"""Minimal baseline SSV training and generation helpers."""

import copy
import gc
import sys

import numpy as np
import torch
from tqdm import tqdm

from SAESTEER.utils import autocast_context, maybe_compile, resolve_precision_dtype


class SSVTrainer:
    """Train a Sparse Steering Vector (SSV) in SAE latent space."""

    def __init__(self, model, sae, layer=20, device="cuda:0"):
        self.model = model
        self.sae = sae
        self.layer = layer
        self.device = device
        self.hook_name = f"blocks.{layer}.hook_resid_post"

        self.model.eval()
        self.sae.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.sae.parameters():
            param.requires_grad_(False)

        self.sae_f32 = copy.deepcopy(sae).cpu()
        for param in self.sae_f32.parameters():
            param.data = param.data.to(torch.float32)
            param.requires_grad_(False)
        self.sae_f32.eval()

        try:
            self.sae_device_dtype = next(self.sae.parameters()).dtype
        except StopIteration:
            self.sae_device_dtype = torch.float32
        self.autocast_dtype = resolve_precision_dtype(device=device)
        self.sae_encode = maybe_compile(self.sae.encode, name="SAE.encode")
        self.sae_decode = maybe_compile(self.sae.decode, name="SAE.decode")

    def _tokenize(self, text):
        return self.model.to_tokens(text).to(self.device)

    def _get_last_token_activation(self, text):
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
        return (
            self.sae_f32.encode(activation.cpu().float().unsqueeze(0))
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

    def _decode_and_steer(self, steered_latent, target_device, target_dtype):
        tensor = torch.tensor(steered_latent, dtype=torch.float32)
        act = self.sae_f32.decode(tensor.unsqueeze(0)).squeeze(0)
        return act.to(target_device, target_dtype)

    def compute_centroids(self, source_texts, target_texts):
        """Compute mean SAE latents for source and target classes."""
        source_latents = []
        for text in tqdm(source_texts, desc="Encoding source texts"):
            act, _ = self._get_last_token_activation(text)
            if act is not None:
                source_latents.append(self._encode(act))

        target_latents = []
        for text in tqdm(target_texts, desc="Encoding target texts"):
            act, _ = self._get_last_token_activation(text)
            if act is not None:
                target_latents.append(self._encode(act))

        if len(source_latents) == 0 or len(target_latents) == 0:
            raise ValueError("Could not encode enough statements to compute centroids.")

        source_centroid = np.mean(np.array(source_latents), axis=0)
        target_centroid = np.mean(np.array(target_latents), axis=0)
        return source_centroid, target_centroid

    def _lm_loss_for_latent(self, steered_latent, source_tokens, target_tokens, source_activation):
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
                token_id = target_tokens[0, t].item()
                if token_id < log_probs.size(0):
                    loss += -log_probs[token_id].item()
                    count += 1
        return loss / count if count > 0 else None

    def _lm_gradient(
        self,
        ssv,
        base_latent,
        source_tokens,
        target_tokens,
        source_activation,
        important_dims,
        base_lm_loss,
        epsilon=1e-4,
    ):
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

    def train(
        self,
        source_texts=None,
        target_texts=None,
        important_dims=None,
        *,
        truthful_texts=None,
        false_texts=None,
        lambda_dist=1.0,
        lambda_reg=0.01,
        lambda_lm=0.5,
        lr=0.01,
        max_iter=100,
        batch_size=32,
        skip_normalization=True,
    ):
        """Train baseline SSV and return (ssv, unnormalized_ssv, initial_ssv, losses)."""
        if source_texts is None or target_texts is None:
            if false_texts is not None and truthful_texts is not None:
                source_texts = false_texts
                target_texts = truthful_texts
            else:
                raise ValueError("source_texts and target_texts must be provided.")
        if important_dims is None or len(important_dims) == 0:
            raise ValueError("important_dims must contain at least one SAE dimension.")

        important_dims = np.asarray(important_dims, dtype=np.int64)
        d_sae = self.sae_f32.cfg.d_sae
        mask = np.zeros(d_sae, dtype=bool)
        mask[important_dims] = True

        print(
            f"Training SSV on {len(important_dims)} selected dimensions "
            f"(lr={lr}, batch_size={batch_size}, max_iter={max_iter})..."
        )

        source_centroid, target_centroid = self.compute_centroids(source_texts, target_texts)

        ssv = np.zeros(d_sae, dtype=np.float32)
        direction = target_centroid - source_centroid
        norm = np.linalg.norm(direction[mask])
        if norm > 0:
            ssv[mask] = direction[mask] / norm
        initial_ssv = ssv.copy()
        print(f"Initial source->target direction norm: {norm:.4f}")

        losses = {"total": [], "distance": [], "lm": [], "reg": []}

        # Stage 2 只优化一条 SSV：靠近目标类 centroid，同时保留语言模型似然和稀疏正则。
        for it in range(max_iter):
            source_idx = np.random.choice(len(source_texts), batch_size, replace=True)
            target_idx = np.random.choice(len(target_texts), batch_size, replace=True)

            dist_loss = 0.0
            lm_loss = 0.0
            dist_grad = np.zeros_like(ssv)
            lm_grad = np.zeros_like(ssv)
            processed = 0

            batch_bar = tqdm(
                range(batch_size),
                desc=f"Iter {it + 1}/{max_iter} batch",
                leave=True,
                mininterval=1.0,
                dynamic_ncols=True,
                file=sys.stdout,
            )
            for i in batch_bar:
                try:
                    source_act, source_tokens = self._get_last_token_activation(
                        source_texts[source_idx[i]]
                    )
                    target_tokens = self._tokenize(target_texts[target_idx[i]])
                    if source_act is None:
                        continue

                    source_latent = self._encode(source_act)
                    steered = source_latent + ssv

                    distance = np.sum((steered - target_centroid) ** 2) - 0.5 * np.sum(
                        (steered - source_centroid) ** 2
                    )
                    dist_loss += distance / batch_size
                    dist_grad += (
                        2 * (steered - target_centroid) - (steered - source_centroid)
                    ) / batch_size

                    with torch.no_grad():
                        base_lm = self._lm_loss_for_latent(
                            steered, source_tokens, target_tokens, source_act
                        )
                        if base_lm is not None:
                            lm_loss += base_lm / batch_size
                            lm_grad += self._lm_gradient(
                                ssv,
                                source_latent,
                                source_tokens,
                                target_tokens,
                                source_act,
                                important_dims,
                                base_lm,
                            ) / batch_size
                            processed += 1

                    if (i + 1) % max(1, batch_size // 8) == 0 or (i + 1) == batch_size:
                        batch_bar.set_postfix(processed=processed)
                except Exception as exc:
                    print(f"Error processing sample: {exc}")

            reg_loss = lambda_reg * np.sum(np.abs(ssv[mask]))
            reg_grad = np.zeros_like(ssv)
            reg_grad[mask] = lambda_reg * np.sign(ssv[mask])

            total_grad = lambda_dist * dist_grad + reg_grad
            if processed > 0:
                total_grad += lambda_lm * lm_grad
            ssv -= lr * total_grad
            ssv[~mask] = 0

            total_loss = lambda_dist * dist_loss + lambda_lm * lm_loss + reg_loss
            losses["total"].append(float(total_loss))
            losses["distance"].append(float(dist_loss))
            losses["lm"].append(float(lm_loss))
            losses["reg"].append(float(reg_loss))

            if (it + 1) % 10 == 0 or it == 0:
                print(
                    f"Iter {it + 1}/{max_iter}  loss={total_loss:.4f}  "
                    f"(dist={dist_loss:.4f}  lm={lm_loss:.4f}  reg={reg_loss:.4f})"
                )

        unnormalized_ssv = ssv.copy()
        if not skip_normalization:
            ssv_norm = np.linalg.norm(ssv)
            if ssv_norm > 0:
                ssv = ssv / ssv_norm
                print(f"Normalized SSV (original norm {ssv_norm:.4f})")
        else:
            print(f"SSV norm: {np.linalg.norm(ssv):.4f}")

        return ssv, unnormalized_ssv, initial_ssv, losses

    def _eos_token_ids(self):
        tokenizer = getattr(self.model, "tokenizer", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eos_ids = set()
        if isinstance(eos_id, (list, tuple, set)):
            eos_ids.update(int(item) for item in eos_id if item is not None)
        elif eos_id is not None:
            eos_ids.add(int(eos_id))

        if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
            for token in ("<|end_of_text|>", "<|eot_id|>", "</s>"):
                token_id = tokenizer.convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id >= 0:
                    eos_ids.add(int(token_id))
        return eos_ids

    def _sample_next_token(self, logits, temperature=0.7, top_p=1.0, banned_token_ids=None):
        if banned_token_ids:
            logits = logits.clone()
            vocab_size = logits.shape[0]
            for token_id in banned_token_ids:
                if 0 <= int(token_id) < vocab_size:
                    logits[int(token_id)] = -torch.inf

        if temperature <= 0:
            return logits.argmax()

        probs = torch.softmax(logits / temperature, dim=0)
        if 0 < top_p < 1:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            remove_mask = cumulative > top_p
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
            probs = torch.zeros_like(probs).scatter(0, sorted_indices, sorted_probs)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                return logits.argmax()
            probs = probs / probs_sum

        return torch.multinomial(probs, num_samples=1).squeeze()

    def generate_steered(
        self,
        text,
        ssv,
        scale=1.0,
        max_new_tokens=50,
        temperature=0.7,
        top_p=1.0,
        min_new_tokens=0,
    ):
        tokens = self.model.to_tokens(text).to(self.device)
        eos_token_ids = self._eos_token_ids()
        min_new_tokens = max(0, min(int(min_new_tokens), int(max_new_tokens)))

        def _hook(act, hook):
            last = act[0, -1, :].clone()
            latent = self._encode(last)
            steered = latent + ssv * scale
            new_act = self._decode_and_steer(steered, last.device, last.dtype)
            act[0, -1, :] = new_act
            return act

        with torch.no_grad():
            current = tokens.clone()
            for step in range(max_new_tokens):
                logits = self.model.run_with_hooks(current, fwd_hooks=[(self.hook_name, _hook)])
                banned_token_ids = eos_token_ids if step < min_new_tokens else None
                next_token = self._sample_next_token(
                    logits[0, -1, :],
                    temperature,
                    top_p,
                    banned_token_ids=banned_token_ids,
                )
                current = torch.cat([current, next_token.view(1, 1)], dim=1)
                if int(next_token.item()) in eos_token_ids:
                    break
        return self.model.to_string(current)

    def generate_baseline(
        self,
        text,
        max_new_tokens=50,
        temperature=0.7,
        top_p=1.0,
        min_new_tokens=0,
    ):
        tokens = self.model.to_tokens(text).to(self.device)
        eos_token_ids = self._eos_token_ids()
        min_new_tokens = max(0, min(int(min_new_tokens), int(max_new_tokens)))
        with torch.no_grad():
            current = tokens.clone()
            for step in range(max_new_tokens):
                logits = self.model(current)
                banned_token_ids = eos_token_ids if step < min_new_tokens else None
                next_token = self._sample_next_token(
                    logits[0, -1, :],
                    temperature,
                    top_p,
                    banned_token_ids=banned_token_ids,
                )
                current = torch.cat([current, next_token.view(1, 1)], dim=1)
                if int(next_token.item()) in eos_token_ids:
                    break
        return self.model.to_string(current)

    def test(
        self,
        ssv,
        test_texts,
        scale_factors=None,
        max_new_tokens=50,
        temperature=0.7,
        top_p=1.0,
        min_new_tokens=0,
    ):
        if scale_factors is None:
            scale_factors = [1.0, 5.0, 10.0]

        results = {float(scale): [] for scale in scale_factors}
        results["baseline"] = []

        for i, text in enumerate(test_texts):
            print(f"\n===== Statement {i + 1}/{len(test_texts)} =====")
            print(f"Input: {text}")

            baseline = self.generate_baseline(
                text,
                max_new_tokens,
                temperature,
                top_p,
                min_new_tokens,
            )
            print(f"\n[Baseline]\n{baseline}")
            results["baseline"].append({"original_input": text, "generated": baseline})

            for scale in scale_factors:
                steered = self.generate_steered(
                    text,
                    ssv,
                    scale,
                    max_new_tokens,
                    temperature,
                    top_p,
                    min_new_tokens,
                )
                print(f"\n[Scale {scale}]\n{steered}")
                results[float(scale)].append({"original_input": text, "generated": steered})

        return results

    def cleanup(self):
        del self.sae_f32
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
