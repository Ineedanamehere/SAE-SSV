"""
Evaluate SAE steering results.

Metrics:
  1. Task-specific evaluation (LLM-as-Judge):
     - politics: Political stance shift (left vs right)
     - truthfulness: Factual accuracy evaluation
     - sentiment: Sentiment/emotion shift (negative vs positive)
  2. Lexical Diversity (ΔMTLD, ΔEntropy)
"""

import os
import sys

# Fix LD_LIBRARY_PATH for vllm (must be before importing vllm)
conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    lib_path = os.path.join(conda_prefix, "lib")
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld}" if current_ld else lib_path
        os.execv(sys.executable, [sys.executable] + sys.argv)


import json
import gc
import torch
from datasets import Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from SAESTEER.evaluator import (
    evaluate_political_shift_batch,
    evaluate_truthfulness_batch,
    evaluate_sentiment_batch,
    compute_diversity_metrics,
    clean_generated_text,
    print_diversity_summary,
)
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))
# ==================== Config ====================
CONFIG = {
    "result_dir": "contrastive_steer_results_sentiment",
    "result_file": "generation_results.json",
    "steered_key": "-6.0",
    "baseline_key": "baseline",
    "judge_model": "Qwen/Qwen3.5-9B",
    "device": "3",  # GPU device ID
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 7000,
    "max_samples": None,
    "batch_size": 50,
    "task_type": "sentiment",  # "politics", "truthfulness", or "sentiment"
}


def load_steering_results(result_dir, result_file, keys):
    """Load and split steering results by source key."""
    path = f"{result_dir}/{result_file}"
    with open(path) as f:
        raw = json.load(f)

    # Flatten to list, handle generated as list or string
    data = []
    for k, records in raw.items():
        for r in records:
            gen = r["generated"]
            if isinstance(gen, list):
                gen = gen[0] if gen else ""
            # Handle both 'input' and 'original_input' field names
            original_input = r.get("original_input") or r.get("input", "")
            data.append({"source_key": str(k), "original_input": original_input, "generated": gen})
    dataset = Dataset.from_list(data)

    # Split by key
    splits = {}
    for key in keys:
        splits[key] = dataset.filter(lambda x: x["source_key"] == key)
        print(f"  {key}: {len(splits[key])} samples")
    return splits


def load_judge_model(cfg):
    """Load evaluation model using vllm."""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["device"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["judge_model"], trust_remote_code=True)
    llm = LLM(
        model=cfg["judge_model"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        max_model_len=cfg["max_model_len"],
    )
    return llm, tokenizer


def create_batch_generate_fn(llm, tokenizer):
    """Create batch generation function for vllm."""
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=512,
    )

    def batch_generate(prompts: list) -> list:
        """Batch generate responses for multiple prompts."""
        # Apply chat template with thinking disabled (for Qwen3)
        formatted = []
        for p in prompts:
            try:
                # Try with enable_thinking=False for Qwen3
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # Fallback for models that don't support enable_thinking
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted.append(text)

        outputs = llm.generate(formatted, sampling_params)
        return [out.outputs[0].text for out in outputs]

    return batch_generate


def extract_texts(steered_ds, baseline_ds):
    """Extract and clean generated texts for diversity analysis."""
    steered, baseline = [], []
    for i in range(min(len(steered_ds), len(baseline_ds))):
        orig = steered_ds[i].get("original_input", "")
        steered.append(clean_generated_text(steered_ds[i].get("generated", ""), orig))
        baseline.append(clean_generated_text(baseline_ds[i].get("generated", ""), orig))
    return steered, baseline


def main():
    cfg = CONFIG

    # 1. Load data
    print("\n[1/4] Loading steering results...")
    datasets = load_steering_results(cfg["result_dir"], cfg["result_file"], [cfg["steered_key"], cfg["baseline_key"]])
    steered_ds = datasets[cfg["steered_key"]]
    baseline_ds = datasets[cfg["baseline_key"]]

    # 2. Load judge model with vllm
    print(f"\n[2/4] Loading judge model ({cfg['judge_model']}) with vllm on GPU {cfg['device']}...")
    llm, tokenizer = load_judge_model(cfg)
    batch_generate_fn = create_batch_generate_fn(llm, tokenizer)

    # 3. LLM-as-Judge evaluation (batch mode)
    task_type = cfg["task_type"]
    if task_type == "truthfulness":
        print(f"\n[3/4] Running truthfulness evaluation (batch mode)...")
        results, summary = evaluate_truthfulness_batch(
            steered_ds, baseline_ds, batch_generate_fn,
            max_samples=cfg["max_samples"],
            batch_size=cfg["batch_size"]
        )
        output_file = "truthfulness_evaluation_results.json"
    elif task_type == "sentiment":
        print(f"\n[3/4] Running sentiment evaluation (batch mode)...")
        results, summary = evaluate_sentiment_batch(
            steered_ds, baseline_ds, batch_generate_fn,
            max_samples=cfg["max_samples"],
            batch_size=cfg["batch_size"]
        )
        output_file = "sentiment_evaluation_results.json"
    else:
        print(f"\n[3/4] Running political stance evaluation (batch mode)...")
        results, summary = evaluate_political_shift_batch(
            steered_ds, baseline_ds, batch_generate_fn,
            max_samples=cfg["max_samples"],
            batch_size=cfg["batch_size"]
        )
        output_file = "political_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump({"summary": summary.to_dict(), "results": results}, f, indent=2)

    # 4. Lexical diversity
    print(f"\n[4/4] Computing lexical diversity...")
    steered_texts, baseline_texts = extract_texts(steered_ds, baseline_ds)
    div_results, div_summary = compute_diversity_metrics(steered_texts, baseline_texts, tokenizer)
    print_diversity_summary(div_summary)
    with open("diversity_evaluation_results.json", "w") as f:
        json.dump({"summary": div_summary, "per_sample": div_results}, f, indent=2)

    # Cleanup
    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
