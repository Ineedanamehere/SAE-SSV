# SAE-SSV
This paper introduces a novel supervised steering approach that operates in sparse, interpretable representation spaces. We employ sparse autoencoders (SAEs) to obtain sparse latent representations that aim to disentangle semantic attributes from model activations. Then we train linear classifiers to identify a small subspace of task-relevant dimensions in latent representations. Finally, we learn supervised steering vectors constrained to this subspace, optimized to align with target behaviors. Experiments across sentiment, truthfulness, and political polarity steering tasks with multiple LLMs demonstrate that our supervised steering vectors achieve higher success rates with minimal degradation in generation quality compared to existing methods. 

**Accepted by EMNLP 2025**

The link of the paper is: https://aclanthology.org/2025.emnlp-main.112.pdf
## Requirements

The code requires the following Python packages:

```txt
python==3.10.19
torch==2.3.1
transformer-lens==1.15.0
psutil==5.9.8
huggingface-hub==0.23.0
sae-lens==0.3.1
datasets==2.20.0
numpy==1.26.4
matplotlib==3.8.3
scikit-learn==1.4.2
tqdm==4.66.4
openai==1.14.0
```
conda create -n sae python=3.10 -y                                                                                                                     
conda activate sae                                                                                                                                     
pip install -r requirements.txt   
## DEMO

The demo implementation of SAE-SSV on Gemma2 and Llama3.1 can be referred to `saessv-demo.ipynb`.

## SAE_SSV Training
### Configuration

Edit the config section in `sae_probe.py`:

```python
DEVICE = "cuda:2"
TARGET_LAYER = 20
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
SPARSITY_THRESHOLD = 1e-1          # L1 threshold for feature selection
steer_scale = [4.0, 5.0, 6.0, -4.0, -5.0, -6.0]
DATASET_TYPE = "sentiment"         # "politics", "truthfulness", or "sentiment"
OUTPUT_DIR = f"probe_results_{DATASET_TYPE}_layer{TARGET_LAYER}"
```
### Usage

```bash
python sae_probe.py
```

### Output Files

| File | Description |
|------|-------------|
| `{dataset}_important_dimensions.npy` | Selected SAE feature indices |
| `{dataset}_ssv_results.pt` | Trained SSV and training metadata |
| `generation_results.json` | Steered generation outputs |

### Class Labels by Dataset

| Dataset | Class A (label=0) | Class B (label=1) |
|---------|-------------------|-------------------|
| `politics` | left | right |
| `truthfulness` | false | true |
| `sentiment` | negative | positive |


## SAE_SSV Evaluation
### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `result_dir` | Path to directory containing `generation_results.json` |
| `steered_key` | JSON key identifying steered outputs (e.g., steering strength) |
| `baseline_key` | JSON key identifying baseline outputs |
| `task_type` | Evaluation type: `"politics"`, `"truthfulness"`, or `"sentiment"` |
| `device` | CUDA device ID (e.g., `"0"`, `"1"`, `"0,1"` for multi-GPU) |
| `max_samples` | Set to integer to limit evaluation samples, `None` for all |

### Basic Usage

```bash
python Evaluation.py
```

## Datasets

We use the following datasets in our experiments:

- **Sentiment**: [Zirui22Ray/sentiment-dataset](https://huggingface.co/datasets/Zirui22Ray/sentiment-dataset)
- **Truthfulness**: [wwbrannon/TruthGen](https://huggingface.co/datasets/wwbrannon/TruthGen)
- **Politics**: [Zirui22Ray/politics-dataset-demo](https://huggingface.co/datasets/Zirui22Ray/politics-dataset-demo)

> [!NOTE]
> This is a preliminary version of the code.  
> We will continue to update it and package it into a more general format as a .py file in the future.


### Citation
If you find the code is valuable, please use this citation.
```
@inproceedings{he2025sae,
  title={Sae-ssv: Supervised steering in sparse representation spaces for reliable control of language models},
  author={He, Zirui and Jin, Mingyu and Shen, Bo and Payani, Ali and Zhang, Yongfeng and Du, Mengnan},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={2207--2236},
  year={2025}
}
```
