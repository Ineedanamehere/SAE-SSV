"""
Minimal single-GPU multi-scale evaluation for simple_repo.

Reads generation_results.json, evaluates each configured scale against baseline,
and writes per-scale judge/diversity files plus final summaries.
"""

import gc
import json
import os
import re
import sys


# ==================== Top-level config ====================
GPU_ID = "0"
EXPERIMENT_TAG = "simple_gemma_sentiment_paper_baseline"
TASK_TYPE = "sentiment"  # "sentiment", "truthfulness", or "politics"
RESULT_FILE = "generation_results.json"
EVAL_SCALES = [4.0, 6.0, -4.0, -6.0]
JUDGE_MODEL = "Qwen/Qwen3.5-9B"
GPU_MEMORY_UTILIZATION = 0.8
MAX_MODEL_LEN = 7000
MAX_SAMPLES = None
BATCH_SIZE = 50

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

conda_prefix = os.environ.get("CONDA_PREFIX", "")

# vLLM 需要在 import 前看到正确的 LD_LIBRARY_PATH。
conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    lib_path = os.path.join(conda_prefix, "lib")
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld}" if current_ld else lib_path
        os.execv(sys.executable, [sys.executable] + sys.argv)


import torch
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from SAESTEER.evaluator import (
    clean_generated_text,
    compute_diversity_metrics,
    evaluate_political_shift_batch,
    evaluate_sentiment_batch,
    evaluate_truthfulness_batch,
    print_diversity_summary,
)
from SAESTEER.utils import setup_environment


def sanitize_experiment_tag(raw_tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_tag.strip())
    return cleaned or "unnamed_experiment"


def sanitize_scale_key(scale_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(scale_key)).replace(".", "_")


def parse_scale_key(scale_key: str):
    try:
        return float(scale_key)
    except (TypeError, ValueError):
        return None


def scale_sort_key(scale_key: str):
    numeric = parse_scale_key(scale_key)
    if numeric is None:
        return (1, str(scale_key))
    return (0, numeric)


EXPERIMENT_TAG = sanitize_experiment_tag(EXPERIMENT_TAG)
TRAIN_RESULT_DIR = os.path.join("runs", EXPERIMENT_TAG, "probe_results")
EVAL_OUTPUT_DIR = os.path.join("runs", EXPERIMENT_TAG, "eval_results")


CONFIG = {
    "experiment_tag": EXPERIMENT_TAG,
    "task_type": TASK_TYPE,
    "result_dir": TRAIN_RESULT_DIR,
    "eval_output_dir": EVAL_OUTPUT_DIR,
    "result_file": RESULT_FILE,
    "baseline_key": "baseline",
    "eval_scales_requested": EVAL_SCALES,
    "eval_scale_policy": "explicit eval_scales",
    "paper_judge_model_deviation": "Qwen/Qwen3.5-9B instead of GPT-4o-mini",
    "judge_model": JUDGE_MODEL,
    "gpu_id": GPU_ID,
    "device": "0",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    "max_model_len": MAX_MODEL_LEN,
    "max_samples": MAX_SAMPLES,
    "batch_size": BATCH_SIZE,
}


def save_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_generation_results(result_dir, result_file):
    path = result_file if os.path.isabs(result_file) else os.path.join(result_dir, result_file)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_requested_scale_keys(raw_results, baseline_key: str, requested_scales):
    numeric_to_key = {}
    skipped = []
    for key in raw_results.keys():
        skey = str(key)
        if skey == baseline_key:
            continue
        numeric = parse_scale_key(skey)
        if numeric is None:
            skipped.append({"scale_key": skey, "reason": "non_numeric"})
            continue
        numeric_to_key.setdefault(numeric, skey)

    selected = []
    missing = []
    seen = set()
    for scale in requested_scales:
        numeric = float(scale)
        if numeric not in numeric_to_key:
            missing.append(numeric)
            continue
        key = numeric_to_key[numeric]
        if key not in seen:
            selected.append(key)
            seen.add(key)

    if missing:
        available = sorted(numeric_to_key.keys())
        raise RuntimeError(
            "Requested eval scales are not present in generation results: "
            f"{missing}. Available numeric scales: {available}"
        )
    selected.sort(key=scale_sort_key)
    return selected, skipped


def load_steering_results(result_dir, result_file, keys):
    raw = load_generation_results(result_dir, result_file)
    data = []
    for key, records in raw.items():
        for record in records:
            generated = record["generated"]
            if isinstance(generated, list):
                generated = generated[0] if generated else ""
            original_input = record.get("original_input") or record.get("input", "")
            data.append(
                {
                    "source_key": str(key),
                    "original_input": original_input,
                    "generated": generated,
                }
            )

    dataset = Dataset.from_list(data)
    splits = {}
    for key in keys:
        splits[key] = dataset.filter(lambda x: x["source_key"] == key)
        print(f"  {key}: {len(splits[key])} samples")
    return splits


def resolve_vllm_gpu_memory_utilization(cfg):
    requested = float(cfg["gpu_memory_utilization"])
    if requested <= 0 or requested > 1:
        raise ValueError(f"GPU_MEMORY_UTILIZATION must be in (0, 1], got {requested}.")
    if not torch.cuda.is_available():
        return requested

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    total_gib = total_bytes / (1024**3)
    free_gib = free_bytes / (1024**3)
    requested_gib = requested * total_gib
    if requested_gib <= free_gib:
        return requested

    margin_gib = 4.0
    adjusted = max(0.0, (free_gib - margin_gib) / total_gib)
    if adjusted < 0.1:
        raise RuntimeError(
            "Not enough free GPU memory to initialize the vLLM judge. "
            f"GPU {cfg['gpu_id']} has {free_gib:.2f}/{total_gib:.2f} GiB free."
        )
    print(
        "Reducing vLLM gpu_memory_utilization from "
        f"{requested:.3f} to {adjusted:.3f}: only "
        f"{free_gib:.2f}/{total_gib:.2f} GiB is free before judge startup."
    )
    return adjusted


def load_judge_model(cfg):
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["judge_model"],
        trust_remote_code=True,
        local_files_only=False,
    )
    llm = LLM(
        model=cfg["judge_model"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=resolve_vllm_gpu_memory_utilization(cfg),
        max_model_len=cfg["max_model_len"],
    )
    return llm, tokenizer


def create_batch_generate_fn(llm, tokenizer):
    sampling_params = SamplingParams(temperature=0.1, max_tokens=512)

    def batch_generate(prompts: list) -> list:
        formatted = []
        for prompt in prompts:
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted.append(text)
        outputs = llm.generate(formatted, sampling_params)
        return [out.outputs[0].text for out in outputs]

    return batch_generate


def extract_texts(steered_ds, baseline_ds):
    steered, baseline = [], []
    for i in range(min(len(steered_ds), len(baseline_ds))):
        original = steered_ds[i].get("original_input", "")
        steered.append(clean_generated_text(steered_ds[i].get("generated", ""), original))
        baseline.append(clean_generated_text(baseline_ds[i].get("generated", ""), original))
    return steered, baseline


def load_training_losses(cfg):
    path = os.path.join(cfg["result_dir"], "training_summary.json")
    if not os.path.exists(path):
        return {
            "path": path,
            "total_last": None,
            "distance_last": None,
            "lm_last": None,
            "reg_last": None,
        }

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    losses = payload.get("losses", {})

    def last_or_none(key):
        values = losses.get(key)
        if not values:
            return None
        return float(values[-1])

    return {
        "path": path,
        "total_last": last_or_none("total"),
        "distance_last": last_or_none("distance"),
        "lm_last": last_or_none("lm"),
        "reg_last": last_or_none("reg"),
    }


def get_task_eval_config(task_type: str):
    if task_type == "truthfulness":
        return evaluate_truthfulness_batch, "truthfulness_evaluation_results"
    if task_type == "sentiment":
        return evaluate_sentiment_batch, "sentiment_evaluation_results"
    return evaluate_political_shift_batch, "political_evaluation_results"


def write_final_summaries(cfg, scale_keys, skipped_scale_keys, per_scale):
    print("\n[5/5] Writing multi-scale summaries...")
    loss_summary = load_training_losses(cfg)
    best_by_sr = max(per_scale, key=lambda item: item["judge_summary"]["success_rate"])

    all_scales_summary = {
        "experiment_tag": cfg["experiment_tag"],
        "task_type": cfg["task_type"],
        "result_file": cfg["result_file"],
        "baseline_key": cfg["baseline_key"],
        "eval_scales_requested": cfg["eval_scales_requested"],
        "eval_scale_policy": cfg["eval_scale_policy"],
        "paper_judge_model_deviation": cfg["paper_judge_model_deviation"],
        "judge_model": cfg["judge_model"],
        "scale_keys": scale_keys,
        "skipped_scale_keys": skipped_scale_keys,
        "num_scales": len(scale_keys),
        "best_by_success_rate": {
            "steered_key": best_by_sr["steered_key"],
            "success_rate": best_by_sr["judge_summary"]["success_rate"],
            "avg_delta_entropy": best_by_sr["diversity_summary"]["avg_delta_entropy"],
        },
        "per_scale": per_scale,
        "loss": loss_summary,
    }
    save_json(os.path.join(cfg["eval_output_dir"], "all_scales_evaluation_summary.json"), all_scales_summary)

    final_metrics = {
        "experiment_tag": cfg["experiment_tag"],
        "task_type": cfg["task_type"],
        "result_file": cfg["result_file"],
        "eval_scales_requested": cfg["eval_scales_requested"],
        "eval_scale_policy": cfg["eval_scale_policy"],
        "paper_judge_model_deviation": cfg["paper_judge_model_deviation"],
        "judge_model": cfg["judge_model"],
        "best_scale": best_by_sr["steered_key"],
        "sr": float(best_by_sr["judge_summary"]["success_rate"]),
        "entropy": {
            "avg_delta_entropy": float(best_by_sr["diversity_summary"]["avg_delta_entropy"]),
            "avg_steered_entropy": float(best_by_sr["diversity_summary"]["avg_steered_entropy"]),
            "avg_baseline_entropy": float(best_by_sr["diversity_summary"]["avg_baseline_entropy"]),
        },
        "loss": loss_summary,
        "all_scales_summary_file": "all_scales_evaluation_summary.json",
    }
    save_json(os.path.join(cfg["eval_output_dir"], "final_sae_ssv_metrics.json"), final_metrics)


def main():
    cfg = CONFIG
    os.makedirs(cfg["eval_output_dir"], exist_ok=True)
    print(f"Experiment tag: {cfg['experiment_tag']}")
    print(f"Reading generation results from: {cfg['result_dir']}")
    print(f"Writing evaluation outputs to: {cfg['eval_output_dir']}")

    print("\n[1/5] Selecting configured steering scales...")
    raw_generation = load_generation_results(cfg["result_dir"], cfg["result_file"])
    scale_keys, skipped_scale_keys = select_requested_scale_keys(
        raw_generation,
        baseline_key=cfg["baseline_key"],
        requested_scales=cfg["eval_scales_requested"],
    )
    if not scale_keys:
        raise RuntimeError("No valid steering scales found for evaluation.")
    print(f"Selected scales for evaluation: {scale_keys}")

    print("\n[2/5] Loading steering results...")
    all_keys = [cfg["baseline_key"], *scale_keys]
    datasets = load_steering_results(cfg["result_dir"], cfg["result_file"], all_keys)
    baseline_ds = datasets[cfg["baseline_key"]]

    print(f"\n[3/5] Loading judge model ({cfg['judge_model']}) on GPU {cfg['gpu_id']}...")
    llm, tokenizer = load_judge_model(cfg)
    batch_generate_fn = create_batch_generate_fn(llm, tokenizer)

    eval_fn, eval_file_prefix = get_task_eval_config(cfg["task_type"])
    per_scale = []

    print("\n[4/5] Evaluating selected scales...")
    for scale_key in scale_keys:
        steered_ds = datasets[scale_key]
        safe_scale_key = sanitize_scale_key(scale_key)
        print(f"\n--- Evaluating scale {scale_key} ---")

        judge_results, summary = eval_fn(
            steered_ds,
            baseline_ds,
            batch_generate_fn,
            max_samples=cfg["max_samples"],
            batch_size=cfg["batch_size"],
        )
        judge_summary = summary.to_dict()
        judge_summary["success_rate"] = float(summary.success_rate)

        judge_filename = f"{eval_file_prefix}__scale_{safe_scale_key}.json"
        save_json(
            os.path.join(cfg["eval_output_dir"], judge_filename),
            {"summary": judge_summary, "results": judge_results},
        )

        steered_texts, baseline_texts = extract_texts(steered_ds, baseline_ds)
        div_results, div_summary = compute_diversity_metrics(steered_texts, baseline_texts, tokenizer)
        print_diversity_summary(div_summary)

        diversity_filename = f"diversity_evaluation_results__scale_{safe_scale_key}.json"
        save_json(
            os.path.join(cfg["eval_output_dir"], diversity_filename),
            {"summary": div_summary, "per_sample": div_results},
        )

        per_scale.append(
            {
                "steered_key": scale_key,
                "steered_scale_float": parse_scale_key(scale_key),
                "num_samples": int(min(len(steered_ds), len(baseline_ds))),
                "judge_summary": judge_summary,
                "diversity_summary": div_summary,
                "output_files": {
                    "judge": judge_filename,
                    "diversity": diversity_filename,
                },
            }
        )

    write_final_summaries(cfg, scale_keys, skipped_scale_keys, per_scale)
    del llm, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nEvaluation complete!")


setup_environment()


if __name__ == "__main__":
    main()
