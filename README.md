# SAE-SSV
This paper introduces a novel supervised steering approach that operates in sparse, interpretable representation spaces. We employ sparse autoencoders (SAEs) to obtain sparse latent representations that aim to disentangle semantic attributes from model activations. Then we train linear classifiers to identify a small subspace of task-relevant dimensions in latent representations. Finally, we learn supervised steering vectors constrained to this subspace, optimized to align with target behaviors. Experiments across sentiment, truthfulness, and political polarity steering tasks with multiple LLMs demonstrate that our supervised steering vectors achieve higher success rates with minimal degradation in generation quality compared to existing methods. 

**Accepted by EMNLP 2025**
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
## Usage

The implementation of SAE-SSV on Gemma2 and Llama3.1 can be referred to `saessv-demo.ipynb`.

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
