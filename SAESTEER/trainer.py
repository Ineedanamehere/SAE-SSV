"""SSV (Sparse Steering Vector) training and testing."""

import os
import copy
import gc

import numpy as np
import torch
from tqdm import tqdm


class SSVTrainer:
    """Train a Sparse Steering Vector (SSV) in SAE latent space.

    The SSV is optimized to steer model representations from one concept
    direction toward another, using three loss terms:
      - Distance loss: push steered latents toward the target centroid
      - LM loss: encourage steered outputs to resemble target text
      - L1 regularization: keep the SSV sparse
    """

    def __init__(self, model, sae, layer=20, device="cuda:0"):
        self.model = model
        self.sae = sae
        self.layer = layer
        self.device = device
        self.hook_name = f"blocks.{layer}.hook_resid_post"

        # Float32 copy of SAE for CPU-side encode/decode
        self.sae_f32 = copy.deepcopy(sae).cpu()
        for param in self.sae_f32.parameters():
            param.data = param.data.to(torch.float32)
        self.sae_f32.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text):
        """Tokenize *text* without running the model."""
        return self.model.to_tokens(text)

    def _get_last_token_activation(self, text):
        """Run model on *text* and return the last-token activation at the target layer."""
        tokens = self._tokenize(text)
        activation = None

        def _hook(act, hook):
            nonlocal activation
            activation = act[0, -1, :].detach().clone()
            return act

        with torch.no_grad():
            self.model.run_with_hooks(tokens, fwd_hooks=[(self.hook_name, _hook)])
        return activation, tokens

    def _encode(self, activation):
        """Encode a single activation vector into SAE latent space (float32, CPU)."""
        return self.sae_f32.encode(
            activation.cpu().float().unsqueeze(0)
        ).squeeze(0).detach().cpu().numpy()

    def _decode_and_steer(self, steered_latent, target_device, target_dtype):
        """Decode a steered latent back to activation space."""
        t = torch.tensor(steered_latent, dtype=torch.float32)
        act = self.sae_f32.decode(t.unsqueeze(0)).squeeze(0)
        return act.to(target_device, target_dtype)

    # ------------------------------------------------------------------
    # Centroid computation
    # ------------------------------------------------------------------

    def compute_centroids(self, truthful_texts, false_texts):
        """Compute mean SAE latent vectors for each class."""
        truthful_latents = []
        for text in tqdm(truthful_texts, desc="Encoding truthful texts"):
            act, _ = self._get_last_token_activation(text)
            if act is not None:
                truthful_latents.append(self._encode(act))

        false_latents = []
        for text in tqdm(false_texts, desc="Encoding false texts"):
            act, _ = self._get_last_token_activation(text)
            if act is not None:
                false_latents.append(self._encode(act))

        if len(truthful_latents) == 0 or len(false_latents) == 0:
            raise ValueError("Could not encode enough statements to compute centroids.")

        truthful_centroid = np.mean(np.array(truthful_latents), axis=0)
        false_centroid = np.mean(np.array(false_latents), axis=0)
        return truthful_centroid, false_centroid

    # ------------------------------------------------------------------
    # LM loss & its numerical gradient
    # ------------------------------------------------------------------

    def _lm_loss_for_latent(self, steered_latent, source_tokens, target_tokens, source_activation):
        """Compute per-token negative log-likelihood of *target_tokens* when
        the model runs on *source_tokens* with the last-token activation
        replaced by the decoded *steered_latent*."""
        steered_act = self._decode_and_steer(
            steered_latent, source_activation.device, source_activation.dtype
        )

        def _hook(act, hook):
            act[0, -1, :] = steered_act
            return act

        output = self.model.run_with_hooks(source_tokens, fwd_hooks=[(self.hook_name, _hook)])

        loss = 0.0
        count = 0
        for t in range(1, min(target_tokens.size(1), 20)):
            if t < output.size(1):
                log_probs = torch.log_softmax(output[0, t - 1, :], dim=0)
                tid = target_tokens[0, t].item()
                if tid < log_probs.size(0):
                    loss += -log_probs[tid].item()
                    count += 1
        return loss / count if count > 0 else None

    def _lm_gradient(self, ssv, base_latent, source_tokens, target_tokens,
                     source_activation, important_dims, base_lm_loss, epsilon=1e-4):
        """Estimate the gradient of LM loss w.r.t. SSV via finite differences
        over *important_dims* only."""
        grad = np.zeros_like(ssv)
        for dim in important_dims:
            perturbed = ssv.copy()
            perturbed[dim] += epsilon
            perturbed_latent = base_latent + perturbed
            perturbed_loss = self._lm_loss_for_latent(
                perturbed_latent, source_tokens, target_tokens, source_activation
            )
            if perturbed_loss is not None:
                grad[dim] = (perturbed_loss - base_lm_loss) / epsilon
        return grad

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, truthful_texts, false_texts, important_dims, *,
              lambda_dist=1.0, lambda_reg=0.01, lambda_lm=0.5,
              lr=0.01, max_iter=200, batch_size=8,
              skip_normalization=True):
        """Train the SSV and return ``(ssv, unnormalized_ssv, initial_ssv, losses)``."""
        d_sae = self.sae_f32.cfg.d_sae
        mask = np.zeros(d_sae, dtype=bool)
        mask[important_dims] = True

        # -- centroids & initialisation --------------------------------
        truthful_centroid, false_centroid = self.compute_centroids(truthful_texts, false_texts)

        ssv = np.zeros(d_sae)
        direction = false_centroid - truthful_centroid
        norm = np.linalg.norm(direction[mask])
        if norm > 0:
            ssv[mask] = direction[mask] / norm
        initial_ssv = ssv.copy()
        print(f"Initial direction norm: {norm:.4f}")

        losses = {"total": [], "distance": [], "lm": [], "reg": []}

        # -- optimisation loop -----------------------------------------
        for it in range(max_iter):
            t_idx = np.random.choice(len(truthful_texts), batch_size, replace=True)
            f_idx = np.random.choice(len(false_texts), batch_size, replace=True)

            dist_loss = 0.0
            lm_loss = 0.0
            dist_grad = np.zeros_like(ssv)
            lm_grad = np.zeros_like(ssv)
            processed = 0

            for i in range(batch_size):
                try:
                    t_act, t_tokens = self._get_last_token_activation(truthful_texts[t_idx[i]])
                    f_tokens = self._tokenize(false_texts[f_idx[i]])
                    if t_act is None:
                        continue

                    t_latent = self._encode(t_act)
                    steered = t_latent + ssv

                    # Distance loss & gradient
                    d = np.sum((steered - false_centroid) ** 2) - 0.5 * np.sum((steered - truthful_centroid) ** 2)
                    dist_loss += d / batch_size
                    dist_grad += (2 * (steered - false_centroid) - (steered - truthful_centroid)) / batch_size

                    # LM loss & gradient (numerical)
                    with torch.no_grad():
                        base_lm = self._lm_loss_for_latent(steered, t_tokens, f_tokens, t_act)
                        if base_lm is not None:
                            lm_loss += base_lm / batch_size
                            lm_grad += self._lm_gradient(
                                ssv, t_latent, t_tokens, f_tokens, t_act,
                                important_dims, base_lm
                            ) / batch_size
                            processed += 1
                except Exception as e:
                    print(f"Error processing sample: {e}")

            # Regularisation
            reg_loss = lambda_reg * np.sum(np.abs(ssv[mask]))
            reg_grad = np.zeros_like(ssv)
            reg_grad[mask] = lambda_reg * np.sign(ssv[mask])

            # Update
            total_grad = lambda_dist * dist_grad + reg_grad
            if processed > 0:
                total_grad += lambda_lm * lm_grad
            ssv -= lr * total_grad
            ssv[~mask] = 0

            total_loss = lambda_dist * dist_loss + lambda_lm * lm_loss + reg_loss
            losses["total"].append(total_loss)
            losses["distance"].append(dist_loss)
            losses["lm"].append(lm_loss)
            losses["reg"].append(reg_loss)

            if (it + 1) % 10 == 0 or it == 0:
                print(f"Iter {it + 1}/{max_iter}  loss={total_loss:.4f}  "
                      f"(dist={dist_loss:.4f}  lm={lm_loss:.4f}  reg={reg_loss:.4f})")

        unnormalized_ssv = ssv.copy()
        if not skip_normalization:
            n = np.linalg.norm(ssv)
            if n > 0:
                ssv = ssv / n
                print(f"Normalized SSV (original norm {n:.4f})")
        else:
            print(f"SSV norm: {np.linalg.norm(ssv):.4f}")

        return ssv, unnormalized_ssv, initial_ssv, losses

    # ------------------------------------------------------------------
    # Testing / generation
    # ------------------------------------------------------------------

    def generate_steered(self, text, ssv, scale=1.0, max_new_tokens=50, temperature=0.7):
        """Generate text with SSV steering applied at every token."""
        tokens = self.model.to_tokens(text)

        def _hook(act, hook):
            last = act[0, -1, :].clone()
            latent = self._encode(last)
            steered = latent + ssv * scale
            new_act = self._decode_and_steer(steered, last.device, last.dtype)
            act[0, -1, :] = new_act
            return act

        with torch.no_grad():
            current = tokens.clone()
            for _ in range(max_new_tokens):
                logits = self.model.run_with_hooks(current, fwd_hooks=[(self.hook_name, _hook)])
                if temperature == 0:
                    next_token = logits[0, -1, :].argmax()
                else:
                    probs = torch.softmax(logits[0, -1, :] / temperature, dim=0)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze()
                current = torch.cat([current, next_token.view(1, 1)], dim=1)
        return self.model.to_string(current)

    def generate_baseline(self, text, max_new_tokens=50, temperature=0.7):
        """Generate text without any steering (baseline)."""
        tokens = self.model.to_tokens(text)
        with torch.no_grad():
            current = tokens.clone()
            for _ in range(max_new_tokens):
                logits = self.model(current)
                if temperature == 0:
                    next_token = logits[0, -1, :].argmax()
                else:
                    probs = torch.softmax(logits[0, -1, :] / temperature, dim=0)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze()
                current = torch.cat([current, next_token.view(1, 1)], dim=1)
        return self.model.to_string(current)

    def test(self, ssv, test_texts, scale_factors=None, max_new_tokens=50):
        """Run baseline + steered generation on a list of test texts."""
        if scale_factors is None:
            scale_factors = [1.0, 5.0, 10.0]

        results = {s: [] for s in scale_factors}
        results["baseline"] = []

        for i, text in enumerate(test_texts):
            print(f"\n===== Statement {i + 1}/{len(test_texts)} =====")
            print(f"Input: {text}")

            # Baseline
            baseline = self.generate_baseline(text, max_new_tokens)
            print(f"\n[Baseline]\n{baseline}")
            results["baseline"].append({"original_input": text, "generated": baseline})

            # Steered
            for scale in scale_factors:
                steered = self.generate_steered(text, ssv, scale, max_new_tokens)
                print(f"\n[Scale {scale}]\n{steered}")
                results[scale].append({"original_input": text, "generated": steered})

        return results

    # ------------------------------------------------------------------
    # Training-free contrastive feature extraction
    # ------------------------------------------------------------------

    def extract_features_without_training(
        self,
        class_a_texts,
        class_b_texts,
        top_k=30,
        verbose=True
    ):
        """
        Extract contrastive steering vector between two classes (training-free).

        Args:
            class_a_texts: List of texts for class A (e.g., left-leaning)
            class_b_texts: List of texts for class B (e.g., right-leaning)
            top_k: Number of top features to select per direction
            verbose: Print progress

        Returns:
            steering_vector (numpy array), class_a_idx, class_b_idx, contrastive_score
        """
        d_sae = self.sae_f32.cfg.d_sae

        # Collect SAE activations
        if verbose:
            print(f"Processing class A ({len(class_a_texts)} samples)...")
        class_a_latents = []
        for text in tqdm(class_a_texts, desc="Class A", disable=not verbose):
            act, _ = self._get_last_token_activation(text)
            if act is not None:
                class_a_latents.append(self._encode(act))

        if verbose:
            print(f"Processing class B ({len(class_b_texts)} samples)...")
        class_b_latents = []
        for text in tqdm(class_b_texts, desc="Class B", disable=not verbose):
            act, _ = self._get_last_token_activation(text)
            if act is not None:
                class_b_latents.append(self._encode(act))

        class_a_acts = np.array(class_a_latents)
        class_b_acts = np.array(class_b_latents)

        # Compute contrastive score
        class_a_mean = class_a_acts.mean(axis=0)
        class_b_mean = class_b_acts.mean(axis=0)
        contrastive_score = class_b_mean - class_a_mean

        # Select top-K features
        class_b_idx = np.argsort(contrastive_score)[-top_k:][::-1]
        class_a_idx = np.argsort(contrastive_score)[:top_k]

        # Build steering vector
        steering_vector = np.zeros(d_sae, dtype=np.float32)
        steering_vector[class_b_idx] = contrastive_score[class_b_idx]
        steering_vector[class_a_idx] = contrastive_score[class_a_idx]


        return steering_vector, class_a_idx, class_b_idx, contrastive_score

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Free the float32 SAE copy."""
        del self.sae_f32
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
