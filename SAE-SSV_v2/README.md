# SAE-SSV v2

This paper introduces a novel supervised steering approach that operates in sparse, interpretable representation spaces. We employ sparse autoencoders (SAEs) to obtain sparse latent representations that aim to disentangle semantic attributes from model activations. Then we train linear classifiers to identify a small subspace of task-relevant dimensions in latent representations. Finally, we learn supervised steering vectors constrained to this subspace, optimized to align with target behaviors. Experiments across sentiment, truthfulness, and political polarity steering tasks with multiple LLMs demonstrate that our supervised steering vectors achieve higher success rates with minimal degradation in generation quality compared to existing methods.

**Accepted by EMNLP 2025**

The link of the paper is: https://aclanthology.org/2025.emnlp-main.112.pdf

## What Changed in v2

`SAE-SSV_v2` keeps the same overall SAE-SSV workflow as the root implementation, but updates the code into a more complete and modular single-GPU pipeline:

- adds a clearer Stage 1 SAE dimension-selection process: F-statistic coarse filtering, latent standardization, multiple L1-regularized probes, probe-weight aggregation, and a separability sweep for selecting `d_steer`;
- organizes the implementation into reusable modules under `SAESTEER/` for datasets, SAE latent extraction, SSV training, evaluation, utilities, and vector-quality helpers;
- writes experiment outputs under `runs/<EXPERIMENT_TAG>/probe_results/` instead of a flat result directory;
- saves additional Stage 1 artifacts such as `selected_indices.npy`, `f_scores.npy`, `difference_vector.npy`, `stage1_probe_stats.json`, and `separability_sweep.json`;
- supports multi-scale generation through `STEER_SCALES`;
- evaluates each steering scale with a local vLLM judge model and diversity metrics, then summarizes all scales in `runs/<EXPERIMENT_TAG>/eval_results/`;
- keeps configuration in the top section of `sae_probe.py` and `Evaluation.py` for easy editing.

## Requirements

The code requires the Python packages pinned in `requirements.txt`.

```bash
conda create -n sae python=3.10 -y
conda activate sae
pip install -r requirements.txt
```

You also need Hugging Face access for the model, SAE, datasets, and local judge model. If required, set `HF_TOKEN` in your environment or in a local `.env` file.

## DEMO

The demo implementation of SAE-SSV on Gemma2 and Llama3.1 can be referred to `saessv-demo.ipynb` in the root directory.

## SAE_SSV Training

### Configuration

Edit the config section in `sae_probe.py`:

```python
GPU_ID = "0"
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

TOP_K_DIMS = 128
STAGE1_NUM_PROBES = 50
STAGE1_SUBSET_FRACTION = 0.5
DSTEER_SELECTION_FRACTION = 0.99

STEER_SCALES = [4.0, 6.0, -4.0, -6.0]
```

### Usage

```bash
cd SAE-SSV_v2
python sae_probe.py
```

### Output Files

Outputs are written to `runs/<EXPERIMENT_TAG>/probe_results/`.

| File | Description |
|------|-------------|
| `paper_selection_summary.json` | Runtime configuration and selection-policy summary |
| `selected_indices.npy` | F-statistic coarse top-k SAE dimensions |
| `f_scores.npy` | F-statistic score for each SAE dimension |
| `difference_vector.npy` | Aggregated signed probe direction in the coarse subspace |
| `{dataset}_important_dimensions.npy` | Final selected SAE dimensions used for SSV training |
| `stage1_probe_stats.json` | Per-probe metrics, selected `d_steer`, standardization stats, and feature order |
| `separability_sweep.json` | Sweep over candidate `d_steer` values |
| `generation_results.json` | Baseline and steered generation outputs |
| `training_summary.json` | Stage 1, Stage 2, loss, norm, and generation metadata |

### Class Labels by Dataset

| Dataset | Class A (label=0) | Class B (label=1) |
|---------|-------------------|-------------------|
| `politics` | left | right |
| `truthfulness` | false | true |
| `sentiment` | negative | positive |

## SAE_SSV Evaluation

### Key Parameters

Edit the config section in `Evaluation.py`:

| Parameter | Description |
|-----------|-------------|
| `EXPERIMENT_TAG` | Experiment name used to locate `runs/<EXPERIMENT_TAG>/probe_results/` |
| `TASK_TYPE` | Evaluation type: `"politics"`, `"truthfulness"`, or `"sentiment"` |
| `RESULT_FILE` | Generation file name, usually `generation_results.json` |
| `EVAL_SCALES` | Steering scales to evaluate |
| `JUDGE_MODEL` | Local vLLM judge model |
| `MAX_SAMPLES` | Set to integer to limit evaluation samples, `None` for all |
| `BATCH_SIZE` | Batch size for judge evaluation |

### Basic Usage

```bash
cd SAE-SSV_v2
python Evaluation.py
```

Evaluation outputs are written to `runs/<EXPERIMENT_TAG>/eval_results/`.

| File | Description |
|------|-------------|
| `{task}_evaluation_results__scale_<scale>.json` | Pairwise LLM-as-judge results for one scale |
| `diversity_evaluation_results__scale_<scale>.json` | MTLD and entropy comparison for one scale |
| `all_scales_evaluation_summary.json` | Cross-scale judge, diversity, and loss summary |
| `final_sae_ssv_metrics.json` | Compact final metrics and best scale by success rate |

## Datasets

We use the following datasets in our experiments:

- **Sentiment**: [Zirui22Ray/sentiment-dataset](https://huggingface.co/datasets/Zirui22Ray/sentiment-dataset)
- **Truthfulness**: [wwbrannon/TruthGen](https://huggingface.co/datasets/wwbrannon/TruthGen)
- **Politics**: [Zirui22Ray/politics-dataset-demo](https://huggingface.co/datasets/Zirui22Ray/politics-dataset-demo)



### Citation

If you find the code is valuable, please use this citation.

```bibtex
@inproceedings{he2025sae,
  title={Sae-ssv: Supervised steering in sparse representation spaces for reliable control of language models},
  author={He, Zirui and Jin, Mingyu and Shen, Bo and Payani, Ali and Zhang, Yongfeng and Du, Mengnan},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={2207--2236},
  year={2025}
}
```
