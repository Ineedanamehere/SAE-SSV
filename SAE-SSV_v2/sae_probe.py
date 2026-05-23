#!/usr/bin/env python3
"""
Minimal SAE-SSV paper_baseline workflow.

This copy keeps the core training path only:
Stage 1 feature selection -> Stage 2 baseline SSV training -> generation.
It intentionally omits multi-GPU launch, checkpoint save/reuse, and token-level
variants so the file can be read from top to bottom like the reference script.
"""

import gc
import json
import os
import re
from datetime import datetime


# ==================== Top-level config ====================
# Edit this block to reproduce an experiment. v2 keeps configuration in-file.
GPU_ID = "0"
# 修改这里即可复现实验；simple_repo 不再提供 CLI/env 包装。
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

MODEL_NAME = "gemma-2-9b"
DATASET_TYPE = "sentiment"  # "sentiment", "truthfulness", or "politics"
EXPERIMENT_TAG = "simple_gemma_sentiment_paper_baseline"

TASK_GEMMA_SAE_LAYERS = {
    "sentiment": 20,
    "truthfulness": 26,
    "politics": 20,
}
TARGET_LAYER = TASK_GEMMA_SAE_LAYERS[DATASET_TYPE]
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = f"layer_{TARGET_LAYER}/width_16k/canonical"
DEVICE = "cuda:0"

TOP_K_DIMS = 128
COARSE_TOP_K = TOP_K_DIMS
DSTEER_MIN = 1
DSTEER_MAX = COARSE_TOP_K
DSTEER_STEP = 1
DSTEER_SELECTION_EPS = 1e-3
DSTEER_SELECTION_FRACTION = 0.99

STAGE1_NUM_PROBES = 50
STAGE1_SUBSET_FRACTION = 0.5
STAGE1_PROBE_EPOCHS = 20
STAGE1_SEED = 42
LEGACY_USE_L1 = True
LEGACY_L1_LAMBDA = 1e-2

STAGE2_LR = 0.01
STAGE2_BATCH_SIZE = 32
STAGE2_MAX_ITER = 100
STAGE2_LAMBDA_DIST = 1.0
STAGE2_LAMBDA_REG = 0.01
STAGE2_LAMBDA_LM = 0.5
STAGE2_SKIP_NORMALIZATION = True

STEER_SCALES = [4.0, 6.0, -4.0, -6.0]
MAX_NEW_TOKENS = 120
MIN_NEW_TOKENS = 0
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 1.0
GENERATION_TEST_SAMPLES = 0  # 0 means all source-class test samples
RANDOM_SEED = 42


import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from SAESTEER.dataset import load_politics, load_sentiment, load_truthfulness
from SAESTEER.extractor import LinearConceptExtractor
from SAESTEER.trainer import SSVTrainer
from SAESTEER.utils import clear_memory, configure_torch_performance, setup_environment


def sanitize_experiment_tag(raw_tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_tag.strip())
    return cleaned or "unnamed_experiment"


EXPERIMENT_TAG = sanitize_experiment_tag(EXPERIMENT_TAG)
OUTPUT_DIR = os.path.join("runs", EXPERIMENT_TAG, "probe_results")


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def runtime_config() -> dict:
    return {
        "model_name": MODEL_NAME,
        "dataset_type": DATASET_TYPE,
        "experiment_tag": EXPERIMENT_TAG,
        "gpu_id": GPU_ID,
        "device": DEVICE,
        "target_layer": TARGET_LAYER,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "pipeline_mode": "paper_baseline",
        "steering_method": "baseline",
        "top_k_dims": TOP_K_DIMS,
        "coarse_top_k": COARSE_TOP_K,
        "dsteer_min": DSTEER_MIN,
        "dsteer_max": DSTEER_MAX,
        "dsteer_step": DSTEER_STEP,
        "dsteer_selection_eps": DSTEER_SELECTION_EPS,
        "dsteer_selection_fraction": DSTEER_SELECTION_FRACTION,
        "stage1_num_probes": STAGE1_NUM_PROBES,
        "stage1_subset_fraction": STAGE1_SUBSET_FRACTION,
        "stage1_probe_epochs": STAGE1_PROBE_EPOCHS,
        "stage1_seed": STAGE1_SEED,
        "stage2_optimizer": "manual_sgd",
        "stage2_lr": STAGE2_LR,
        "stage2_batch_size": STAGE2_BATCH_SIZE,
        "stage2_max_iter": STAGE2_MAX_ITER,
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_new_tokens": MIN_NEW_TOKENS,
        "generation_temperature": GENERATION_TEMPERATURE,
        "generation_top_p": GENERATION_TOP_P,
        "generation_test_samples": GENERATION_TEST_SAMPLES,
        "steer_scales": STEER_SCALES,
        "checkpoint_policy": "disabled: no .pt checkpoint is saved or loaded in simple_repo",
    }


def load_task_datasets(task_type: str):
    if task_type == "politics":
        train_dataset, test_dataset, target_train, source_train = load_politics()
        source_name, target_name = "left", "right"
    elif task_type == "truthfulness":
        train_dataset, test_dataset, target_train, source_train = load_truthfulness()
        source_name, target_name = "false", "true"
    elif task_type == "sentiment":
        train_dataset, test_dataset, target_train, source_train = load_sentiment()
        source_name, target_name = "negative", "positive"
    else:
        raise ValueError(f"Unsupported dataset type: {task_type}")
    return train_dataset, test_dataset, source_train, target_train, source_name, target_name


def load_model_and_sae():
    print(f"Loading model: {MODEL_NAME} on {DEVICE} ...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        torch_dtype=RUNTIME_DTYPE,
    )
    print("Model loaded.")

    print(f"Loading SAE: {SAE_RELEASE} / {SAE_ID} ...")
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
    sae = sae.to(DEVICE)
    print("SAE loaded.")
    return model, sae


def precompute_latents_as_arrays(extractor: LinearConceptExtractor, dataset):
    print("\n========== STAGE 1: Precompute train latents ==========")
    records = extractor.precompute_latents_for_indices(
        text_dataset=dataset,
        indices=list(range(len(dataset["text"]))),
        batch_size=4,
    )
    records = sorted(records, key=lambda item: item[0])
    latents = np.stack([np.asarray(latent, dtype=np.float32) for _, latent, _ in records])
    labels = np.asarray([label for _, _, label in records], dtype=np.int64)
    return latents, labels


def run_stage1(extractor: LinearConceptExtractor, train_latents, train_labels):
    print("\n========== STAGE 1: Paper-aligned feature selection ==========")
    print(
        "先用 F-stat 做 coarse top-k，是为了把全 SAE 空间收缩到最可能区分类别的子空间；"
        "随后再训练多个 probes，降低单次随机划分带来的不稳定性。"
    )

    selected_latents, selected_indices, f_scores = extractor.select_features_by_f_statistic(
        train_latents,
        train_labels,
        top_k=COARSE_TOP_K,
    )
    standardized_latents, standardization_mean, standardization_scale = extractor.standardize_latents(
        selected_latents
    )

    classifiers, importance_probe_vectors, direction_probe_vectors, probe_stats = (
        extractor.train_multiple_linear_classifiers(
            latents=standardized_latents,
            labels=train_labels,
            num_probes=STAGE1_NUM_PROBES,
            subset_fraction=STAGE1_SUBSET_FRACTION,
            seed=STAGE1_SEED,
            val_size=0.0,
            batch_size=32,
            num_epochs=STAGE1_PROBE_EPOCHS,
            lr=1e-4,
            weight_decay=5e-2,
            use_l1=LEGACY_USE_L1,
            lambda_l1=LEGACY_L1_LAMBDA,
        )
    )

    # 多个 probe 的权重平均后作为稳定的重要性估计和方向估计。
    importance_scores, concept_direction = extractor.aggregate_probe_vectors(
        importance_probe_vectors,
        direction_probe_vectors,
    )

    print(
        "Separability sweep 会从 top-1 到 top-k 搜索；这里选择达到最佳分离度 "
        f"{DSTEER_SELECTION_FRACTION:.0%} 的最小 d_steer。"
    )
    important_dims, best_separation, separability_sweep, feature_order = (
        extractor.select_dims_by_separability(
            standardized_latents,
            train_labels,
            selected_indices,
            importance_scores,
            concept_direction,
            d_min=DSTEER_MIN,
            d_max=min(DSTEER_MAX, len(selected_indices)),
            d_step=DSTEER_STEP,
            separation_tolerance=DSTEER_SELECTION_EPS,
            separation_fraction=DSTEER_SELECTION_FRACTION,
        )
    )

    if len(important_dims) == 0:
        raise RuntimeError("No SAE dimensions selected by separability sweep.")

    stage1_train_acc = float(np.mean([p["test_accuracy"] for p in probe_stats]))
    stage1_train_f1 = float(np.mean([p["test_f1"] for p in probe_stats]))
    model_info = {
        "model_name": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "input_dim": int(selected_latents.shape[1]),
        "target_layer": TARGET_LAYER,
        "num_probes": STAGE1_NUM_PROBES,
        "subset_fraction": STAGE1_SUBSET_FRACTION,
        "coarse_selection_method": "f_statistic_topk",
        "coarse_top_k": COARSE_TOP_K,
        "aggregation_method": "mean_abs_probe_vector",
        "stage1_latent_standardization": "zscore_after_f_stat_topk",
        "direction_source": "positive_class_weight",
        "dsteer_selection_method": "best_fraction_smallest_d",
        "d_steer": int(len(important_dims)),
        "best_separation": best_separation,
        "test_accuracy_mean": stage1_train_acc,
        "test_f1_mean": stage1_train_f1,
        "test_accuracy_std": float(np.std([p["test_accuracy"] for p in probe_stats])),
        "test_f1_std": float(np.std([p["test_f1"] for p in probe_stats])),
    }

    np.save(os.path.join(OUTPUT_DIR, "selected_indices.npy"), selected_indices)
    np.save(os.path.join(OUTPUT_DIR, "f_scores.npy"), f_scores)
    np.save(os.path.join(OUTPUT_DIR, "difference_vector.npy"), concept_direction)
    np.save(os.path.join(OUTPUT_DIR, f"{DATASET_TYPE}_important_dimensions.npy"), important_dims)
    save_json(
        os.path.join(OUTPUT_DIR, "stage1_probe_stats.json"),
        {
            "num_probes": STAGE1_NUM_PROBES,
            "subset_fraction": STAGE1_SUBSET_FRACTION,
            "seed": STAGE1_SEED,
            "coarse_top_k": COARSE_TOP_K,
            "d_steer": int(len(important_dims)),
            "best_separation": best_separation,
            "probe_stats": probe_stats,
            "standardization_mean": standardization_mean,
            "standardization_scale": standardization_scale,
            "feature_order": feature_order,
            "model_info": model_info,
        },
    )
    save_json(
        os.path.join(OUTPUT_DIR, "separability_sweep.json"),
        {
            "d_min": DSTEER_MIN,
            "d_max": int(min(DSTEER_MAX, len(selected_indices))),
            "d_step": DSTEER_STEP,
            "selection_eps": DSTEER_SELECTION_EPS,
            "selection_fraction": DSTEER_SELECTION_FRACTION,
            "best": best_separation,
            "sweep": separability_sweep,
        },
    )
    return important_dims, best_separation, model_info


def run_stage2_and_generation(
    trainer: SSVTrainer,
    important_dims,
    source_train,
    target_train,
    test_dataset,
    source_name,
):
    print("\n========== STAGE 2: Baseline SSV training ==========")
    start_time = datetime.now()
    ssv, unnormalized_ssv, initial_ssv, losses = trainer.train(
        source_texts=source_train,
        target_texts=target_train,
        important_dims=important_dims,
        lambda_dist=STAGE2_LAMBDA_DIST,
        lambda_reg=STAGE2_LAMBDA_REG,
        lambda_lm=STAGE2_LAMBDA_LM,
        lr=STAGE2_LR,
        max_iter=STAGE2_MAX_ITER,
        batch_size=STAGE2_BATCH_SIZE,
        skip_normalization=STAGE2_SKIP_NORMALIZATION,
    )
    training_time = datetime.now() - start_time
    print(f"Final training completed in: {training_time}")

    print("\n========== GENERATION ==========")
    source_test_texts = [item["text"] for item in test_dataset if item["label"] == 0]
    if GENERATION_TEST_SAMPLES > 0:
        source_test_texts = source_test_texts[:GENERATION_TEST_SAMPLES]
    print(f"Generating from {len(source_test_texts)} {source_name} test samples...")
    generation_results = trainer.test(
        ssv=ssv,
        test_texts=source_test_texts,
        scale_factors=STEER_SCALES,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        min_new_tokens=MIN_NEW_TOKENS,
    )
    save_json(os.path.join(OUTPUT_DIR, "generation_results.json"), generation_results)
    return {
        "losses": losses,
        "training_time": str(training_time),
        "ssv_norm": float(np.linalg.norm(ssv)),
        "unnormalized_ssv_norm": float(np.linalg.norm(unnormalized_ssv)),
        "initial_ssv_norm": float(np.linalg.norm(initial_ssv)),
        "generated_sample_count": int(len(source_test_texts)),
    }


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Experiment tag: {EXPERIMENT_TAG}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Dataset: {DATASET_TYPE}")
    print("Checkpoint save/reuse is disabled in simple_repo; only JSON/NPY artifacts are written.")

    print(f"\nLoading dataset: {DATASET_TYPE}...")
    train_dataset, test_dataset, source_train, target_train, source_name, target_name = (
        load_task_datasets(DATASET_TYPE)
    )
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"Source ({source_name}): {len(source_train)}, Target ({target_name}): {len(target_train)}")

    save_json(
        os.path.join(OUTPUT_DIR, "paper_selection_summary.json"),
        {
            "selection_source": "paper_aligned_f_stat_probe_separability",
            **runtime_config(),
        },
    )

    model, sae = load_model_and_sae()
    extractor = LinearConceptExtractor(
        sae=sae,
        language_model=model,
        target_layer=TARGET_LAYER,
        device=DEVICE,
        use_l1=LEGACY_USE_L1,
        lambda_l1=LEGACY_L1_LAMBDA,
    )
    trainer = SSVTrainer(model, sae, layer=TARGET_LAYER, device=DEVICE)

    train_latents, train_labels = precompute_latents_as_arrays(extractor, train_dataset)
    important_dims, best_separation, model_info = run_stage1(extractor, train_latents, train_labels)
    stage2_summary = run_stage2_and_generation(
        trainer,
        important_dims,
        source_train,
        target_train,
        test_dataset,
        source_name,
    )

    save_json(
        os.path.join(OUTPUT_DIR, "training_summary.json"),
        {
            **runtime_config(),
            "source_class": source_name,
            "target_class": target_name,
            "d_steer": int(len(important_dims)),
            "important_dims": important_dims,
            "best_separation": best_separation,
            "model_info": model_info,
            **stage2_summary,
        },
    )

    trainer.cleanup()
    del trainer, extractor, model, sae
    gc.collect()
    clear_memory()
    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("PIPELINE COMPLETE!")


setup_environment()
clear_memory()
RUNTIME_DTYPE = configure_torch_performance(DEVICE)
torch.set_grad_enabled(True)


if __name__ == "__main__":
    main()
