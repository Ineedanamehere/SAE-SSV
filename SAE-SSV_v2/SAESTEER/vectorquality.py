"""Steering vector quality evaluation."""

import numpy as np
import torch
from tqdm import tqdm


def evaluate_steering_vector(
    model,
    sae,
    steering_vector,
    class_a_texts,
    class_b_texts,
    target_layer=20,
    verbose=True
):
    """
    Evaluate steering vector's classification accuracy using projection scores.

    Args:
        model: HookedTransformer model
        sae: SAE model
        steering_vector: The steering vector to evaluate (numpy array)
        class_a_texts: Test texts for class A (expected negative scores)
        class_b_texts: Test texts for class B (expected positive scores)
        target_layer: Layer to hook for activations
        verbose: Print results

    Returns:
        dict with scores, threshold, accuracy, and separation
    """
    hook_name = f"blocks.{target_layer}.hook_resid_post"

    def _score_texts(texts, desc=""):
        scores = []
        for text in tqdm(texts, desc=desc, disable=not verbose):
            activation = None

            def _hook(act, hook):
                nonlocal activation
                activation = act[0, -1, :].detach().clone()
                return act

            tokens = model.to_tokens(text)
            with torch.no_grad():
                model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])
                sae_act = sae.encode(activation.unsqueeze(0)).squeeze(0).cpu().float().numpy()

            scores.append(np.dot(sae_act, steering_vector))
            del activation, tokens

        return np.array(scores)

    # Score both classes
    class_a_scores = _score_texts(class_a_texts, "Scoring class A")
    class_b_scores = _score_texts(class_b_texts, "Scoring class B")

    # Compute metrics
    threshold = (class_a_scores.mean() + class_b_scores.mean()) / 2.0
    class_a_correct = (class_a_scores < threshold).sum()
    class_b_correct = (class_b_scores >= threshold).sum()
    accuracy = (class_a_correct + class_b_correct) / (len(class_a_scores) + len(class_b_scores))
    separation = class_b_scores.mean() - class_a_scores.mean()

    if verbose:
        print(f"\nProjection scores:")
        print(f"  Class A — mean={class_a_scores.mean():.4f}, std={class_a_scores.std():.4f}")
        print(f"  Class B — mean={class_b_scores.mean():.4f}, std={class_b_scores.std():.4f}")
        print(f"  Separation (B - A) = {separation:.4f}")
        print(f"  Accuracy (threshold={threshold:.4f}): {accuracy:.2%}")

    return {
        "class_a_scores": class_a_scores,
        "class_b_scores": class_b_scores,
        "threshold": threshold,
        "accuracy": accuracy,
        "separation": separation
    }
