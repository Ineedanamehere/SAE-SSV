#!/usr/bin/env python3
"""
SAE-SSV Pipeline: Concept Vector Extraction + SSV Training & Testing
Merged from saessv_step1.py and saessv_step2.py.
"""

import os
import json
import torch
import numpy as np
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub import login
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datetime import datetime

from SAESTEER.extractor import LinearConceptExtractor, analyze_truthfulness_with_concept, test_concept_vector_difference
from SAESTEER.trainer import SSVTrainer
from SAESTEER.dataset import load_politics, load_truthfulness, load_sentiment
from SAESTEER.utils import clear_memory

# ==================== Config ====================
DEVICE = "cuda:2"
TARGET_LAYER = 20
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
SPARSITY_THRESHOLD = 1e-2
steer_scale = [4.0, 6.0, -4.0, -6.0]
# Dataset selection: "politics", "truthfulness", or "sentiment"
DATASET_TYPE = "sentiment"
OUTPUT_DIR = f"probe_results_{DATASET_TYPE}_layer{TARGET_LAYER}"

# ==================== Setup ====================
load_dotenv()
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

clear_memory()
torch.set_grad_enabled(True)

# ==================== Load Model ====================
print("Loading model...")
model = HookedTransformer.from_pretrained(
    "gemma-2-9b",
    device=DEVICE,
    torch_dtype=torch.bfloat16,
)
print("Model loaded.")

# ==================== Load SAE ====================
print("Loading SAE...")
sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
sae = sae.to(DEVICE)
print("SAE loaded.")

# ==================== Load Dataset ====================
print(f"Loading dataset: {DATASET_TYPE}...")
if DATASET_TYPE == "politics":
    train_dataset, test_dataset, class_b_train, class_a_train = load_politics()
    CLASS_A_NAME, CLASS_B_NAME = "left", "right"
elif DATASET_TYPE == "truthfulness":
    train_dataset, test_dataset, class_b_train, class_a_train = load_truthfulness()
    CLASS_A_NAME, CLASS_B_NAME = "false", "true"
elif DATASET_TYPE == "sentiment":
    train_dataset, test_dataset, class_b_train, class_a_train = load_sentiment()
    CLASS_A_NAME, CLASS_B_NAME = "negative", "positive"
else:
    raise ValueError(f"Unsupported dataset type: {DATASET_TYPE}")

print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
print(f"Class A ({CLASS_A_NAME}): {len(class_a_train)}, Class B ({CLASS_B_NAME}): {len(class_b_train)}")

# ==================== Step 1: Extract Concept Vectors ====================
print("\n========== STEP 1: Extracting concept vectors ==========")
extractor = LinearConceptExtractor(
    sae=sae,
    language_model=model,
    target_layer=TARGET_LAYER,
    device=DEVICE,
    use_l1=True,
    lambda_l1=1e-2
)

results, classifier = extractor.extract_concept_vectors(
    text_dataset=train_dataset,
    output_dir=OUTPUT_DIR
)

selected_indices = results['selected_indices']
difference_vector = results['difference_vector']

# Test set evaluation
left_scores, right_scores, separation, accuracy = test_concept_vector_difference(
    test_dataset.select(range(100)), difference_vector, selected_indices, sae, model, TARGET_LAYER
)
print("Step 1 complete.")

# ==================== Step 2: Extract Key Dimensions ====================
print("\n========== STEP 2: SSV Training & Testing ==========")
print("Extracting key dimensions via L1 sparsity threshold...")
importance = np.abs(difference_vector)
important_mask = importance > SPARSITY_THRESHOLD
important_reduced_indices = np.where(important_mask)[0]
top_original_indices = np.array(selected_indices)[important_reduced_indices]

print(f"Threshold: {SPARSITY_THRESHOLD}")
print(f"Selected {len(top_original_indices)} / {len(difference_vector)} dimensions")
for i, idx in enumerate(top_original_indices):
    print(f"  {i+1}: SAE index {idx} (|w|={importance[important_reduced_indices[i]]:.4f})")

np.save(os.path.join(OUTPUT_DIR, f"{DATASET_TYPE}_important_dimensions.npy"), top_original_indices)

# ==================== Train SSV ====================
print("\nTraining SSV...")
print(f"Class B ({CLASS_B_NAME}): {len(class_b_train)}, Class A ({CLASS_A_NAME}): {len(class_a_train)}")

trainer = SSVTrainer(model, sae, layer=TARGET_LAYER, device=DEVICE)

start_time = datetime.now()
ssv, unnormalized_ssv, initial_ssv, losses = trainer.train(
    truthful_texts=class_b_train,  # Target class (right/true/positive)
    false_texts=class_a_train,      # Source class (left/false/negative)
    important_dims=top_original_indices,
    lambda_dist=1.0,
    lambda_reg=0.01,
    lambda_lm=0.5,
    lr=0.01,
    max_iter=100,
    batch_size=32,
    skip_normalization=True,
)
training_time = datetime.now() - start_time
print(f"Training completed in: {training_time}")

torch.save({
    'ssv': ssv,
    'unnormalized_ssv': unnormalized_ssv,
    'losses': losses,
    'important_dims': top_original_indices,
    'training_time': str(training_time),
    'dataset_type': DATASET_TYPE,
    'class_a': CLASS_A_NAME,
    'class_b': CLASS_B_NAME,
}, os.path.join(OUTPUT_DIR, f"{DATASET_TYPE}_ssv_results.pt"))
print(f"SSV saved to {OUTPUT_DIR}/{DATASET_TYPE}_ssv_results.pt")

# ==================== Test SSV ====================
print(f"\nTesting SSV on {CLASS_A_NAME} test samples...")
class_a_test = [item['text'] for item in test_dataset if item['label'] == 0]
print(f"Test samples: {len(class_a_test)}")

test_results = trainer.test(
    ssv=ssv,
    test_texts=class_a_test,
    scale_factors=steer_scale,
    max_new_tokens=120,
)

with open(os.path.join(OUTPUT_DIR, "generation_results.json"), "w") as f:
    json.dump(test_results, f)
print("Test results saved.")

trainer.cleanup()

# ==================== Done ====================
print(f"\nAll results saved to {OUTPUT_DIR}/")
print("PIPELINE COMPLETE!")
