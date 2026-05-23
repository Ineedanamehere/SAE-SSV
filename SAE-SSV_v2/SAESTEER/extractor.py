"""Concept vector extraction via linear probes on SAE latent space."""

import os
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

from SAESTEER.utils import autocast_context, dataloader_kwargs, maybe_compile, resolve_precision_dtype


class LinearConceptExtractor:
    """Legacy Stage 1 extractor: non-zero features + single linear probe."""

    def __init__(
        self,
        sae,
        language_model,
        target_layer=None,
        device="cuda",
        use_l1=False,
        lambda_l1=1e-3,
    ):
        self.sae = sae
        self.language_model = language_model
        self.target_layer = (
            target_layer if target_layer is not None else language_model.cfg.n_layers - 1
        )
        self.device = device
        self.sae.to(device)
        self.d_sae = sae.cfg.d_sae
        self.use_l1 = use_l1
        self.lambda_l1 = lambda_l1
        self.autocast_dtype = resolve_precision_dtype(device=device)
        self.sae_encode = maybe_compile(self.sae.encode, name="SAE.encode")
        print(
            f"Initializing LinearConceptExtractor, using model layer "
            f"{self.target_layer + 1}/{language_model.cfg.n_layers}"
        )

    def precompute_latents_for_indices(self, text_dataset, indices, batch_size=16):
        print(f"Precomputing latent representations for {len(indices)} assigned samples...")
        all_records = []
        texts = text_dataset["text"]
        labels = text_dataset["label"]
        for start in tqdm(range(0, len(indices), batch_size)):
            batch_indices = indices[start : start + batch_size]
            for dataset_index in batch_indices:
                text = texts[dataset_index]
                label = labels[dataset_index]
                tokens = self.language_model.to_tokens(text).to(self.device)
                with torch.no_grad():
                    _, cache = self.language_model.run_with_cache(
                        tokens,
                        stop_at_layer=self.target_layer + 1,
                        names_filter=lambda name: name
                        == f"blocks.{self.target_layer}.hook_resid_post",
                    )
                    token_residual = cache["resid_post", self.target_layer][0, -1, :]
                    with autocast_context(self.device, self.autocast_dtype):
                        latent = self.sae_encode(token_residual.unsqueeze(0)).squeeze(0)
                    latent_np = latent.to(torch.float32).detach().cpu().numpy()
                    all_records.append((int(dataset_index), latent_np, int(label)))
                del cache
        return all_records

    def precompute_latents(self, text_dataset, batch_size=16):
        print(
            f"Precomputing latent representations, total {len(text_dataset['text'])} samples..."
        )
        records = self.precompute_latents_for_indices(
            text_dataset,
            list(range(len(text_dataset["text"]))),
            batch_size=batch_size,
        )
        all_latents = [torch.tensor(latent, dtype=torch.float32) for _, latent, _ in records]
        all_labels = [label for _, _, label in records]
        return all_latents, all_labels

    def select_important_features(self, latents):
        """Remove all-zero columns and keep only non-zero features."""
        print("Performing feature selection, removing all-zero columns...")
        if isinstance(latents[0], torch.Tensor):
            latents_np = np.array([l.cpu().numpy() for l in latents])
        else:
            latents_np = np.array(latents)

        n_features = latents_np.shape[1]
        non_zero_mask = ~np.all(latents_np == 0, axis=0)
        selected_indices = np.where(non_zero_mask)[0]
        selected_latents = latents_np[:, selected_indices]

        n_zero_cols = n_features - len(selected_indices)
        print(
            f"All-zero columns: {n_zero_cols}/{n_features} | "
            f"Active columns: {len(selected_indices)}"
        )
        print(
            f"Feature selection completed, reduced from {n_features} "
            f"to {len(selected_indices)} features"
        )
        return selected_latents, selected_indices

    def compute_f_statistics(self, latents, labels) -> np.ndarray:
        """Compute binary-class ANOVA F-statistics for each SAE latent dimension."""
        latents_np = np.asarray(latents, dtype=np.float32)
        labels_np = np.asarray(labels)
        if latents_np.ndim != 2:
            raise ValueError("latents must have shape [num_samples, num_features]")
        if latents_np.shape[0] != labels_np.shape[0]:
            raise ValueError("latents and labels must have the same number of rows")

        class0 = latents_np[labels_np == 0]
        class1 = latents_np[labels_np == 1]
        if class0.size == 0 or class1.size == 0:
            raise ValueError("Both classes must be present to compute F-statistics")

        mean0 = class0.mean(axis=0)
        mean1 = class1.mean(axis=0)
        overall = latents_np.mean(axis=0)
        n0 = class0.shape[0]
        n1 = class1.shape[0]
        n_total = latents_np.shape[0]

        between = n0 * (mean0 - overall) ** 2 + n1 * (mean1 - overall) ** 2
        var0 = class0.var(axis=0, ddof=1) if n0 > 1 else np.zeros(latents_np.shape[1], dtype=np.float32)
        var1 = class1.var(axis=0, ddof=1) if n1 > 1 else np.zeros(latents_np.shape[1], dtype=np.float32)
        within = ((max(n0 - 1, 0) * var0) + (max(n1 - 1, 0) * var1)) / max(n_total - 2, 1)
        f_scores = between / (within + 1e-8)
        return np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def select_features_by_f_statistic(self, latents, labels, top_k=128):
        """Select the top-k SAE dimensions by binary-class F-statistic."""
        print(f"Selecting coarse SAE subspace by F-statistic (top_k={top_k})...")
        latents_np = np.asarray(latents, dtype=np.float32)
        f_scores = self.compute_f_statistics(latents_np, labels)
        active_mask = ~np.all(latents_np == 0, axis=0)
        ranked = np.argsort(-f_scores)
        ranked = ranked[active_mask[ranked]]
        if ranked.size == 0:
            raise ValueError("No active SAE dimensions found for F-statistic selection")
        k = max(1, min(int(top_k), ranked.size))
        selected_indices = ranked[:k].astype(np.int64)
        selected_latents = latents_np[:, selected_indices]
        print(
            f"F-statistic selection completed, reduced from {latents_np.shape[1]} "
            f"to {len(selected_indices)} features"
        )
        return selected_latents, selected_indices, f_scores

    def standardize_latents(self, latents, eps=1e-6):
        """Z-score SAE latents columnwise for Stage 1 probe training."""
        latents_np = np.asarray(latents, dtype=np.float32)
        if latents_np.ndim != 2:
            raise ValueError("latents must have shape [num_samples, num_features]")
        mean = latents_np.mean(axis=0, keepdims=True)
        scale = latents_np.std(axis=0, keepdims=True)
        safe_scale = np.where(scale > eps, scale, 1.0).astype(np.float32)
        standardized = (latents_np - mean) / safe_scale
        return (
            standardized.astype(np.float32),
            mean.squeeze(0).astype(np.float32),
            safe_scale.squeeze(0).astype(np.float32),
        )

    def select_dims_by_separability(
        self,
        latents,
        labels,
        selected_indices,
        importance,
        signed_vector,
        *,
        d_min=1,
        d_max=None,
        d_step=1,
        separation_tolerance=0.0,
        separation_fraction=None,
    ):
        """Choose the smallest top-d subspace that nearly matches best separation."""
        latents_np = np.asarray(latents, dtype=np.float32)
        labels_np = np.asarray(labels)
        selected_indices_np = np.asarray(selected_indices, dtype=np.int64)
        importance_np = np.asarray(importance, dtype=np.float32)
        signed_vector_np = np.asarray(signed_vector, dtype=np.float32)

        if latents_np.shape[1] != selected_indices_np.shape[0]:
            raise ValueError("latents columns must match selected_indices length")
        if importance_np.shape[0] != selected_indices_np.shape[0]:
            raise ValueError("importance length must match selected_indices length")
        if signed_vector_np.shape[0] != selected_indices_np.shape[0]:
            raise ValueError("signed_vector length must match selected_indices length")

        d_min = max(1, int(d_min))
        d_max = selected_indices_np.shape[0] if d_max is None else int(d_max)
        d_max = max(d_min, min(d_max, selected_indices_np.shape[0]))
        d_step = max(1, int(d_step))
        separation_tolerance = max(0.0, float(separation_tolerance))
        if separation_fraction is not None:
            separation_fraction = min(1.0, max(0.0, float(separation_fraction)))

        order = np.argsort(-importance_np)
        row_norms = np.linalg.norm(latents_np, axis=1, keepdims=True)
        normalized_latents = latents_np / np.maximum(row_norms, 1e-8)

        sweep = []
        best = None
        for d in range(d_min, d_max + 1, d_step):
            top_local = order[:d]
            direction = np.zeros_like(signed_vector_np)
            direction[top_local] = signed_vector_np[top_local]
            direction_norm = np.linalg.norm(direction)
            if direction_norm <= 0:
                separation = 0.0
                class0_mean = 0.0
                class1_mean = 0.0
            else:
                scores = normalized_latents @ (direction / direction_norm)
                class0_scores = scores[labels_np == 0]
                class1_scores = scores[labels_np == 1]
                class0_mean = float(class0_scores.mean()) if class0_scores.size else 0.0
                class1_mean = float(class1_scores.mean()) if class1_scores.size else 0.0
                separation = class1_mean - class0_mean

            item = {
                "d": int(d),
                "separation": float(separation),
                "class0_mean": float(class0_mean),
                "class1_mean": float(class1_mean),
            }
            sweep.append(item)
            if best is None or separation > best["separation"]:
                best = item

        if best is None:
            raise RuntimeError("Could not select dimensions by separability")

        if separation_fraction is not None and best["separation"] > 0:
            threshold = best["separation"] * separation_fraction
            selection_method = "best_fraction_smallest_d"
        else:
            threshold = best["separation"] - separation_tolerance
            selection_method = "best_minus_eps_smallest_d"

        selected = next(item for item in sweep if item["separation"] >= threshold)
        selected = {
            **selected,
            "selection_method": selection_method,
            "selection_threshold": float(threshold),
            "selection_fraction": (
                None if separation_fraction is None else float(separation_fraction)
            ),
            "selection_tolerance": float(separation_tolerance),
            "global_best_d": int(best["d"]),
            "global_best_separation": float(best["separation"]),
            "global_best_class0_mean": float(best["class0_mean"]),
            "global_best_class1_mean": float(best["class1_mean"]),
        }

        selected_d = int(selected["d"])
        best_local = order[:selected_d]
        important_dims = selected_indices_np[best_local]
        return important_dims, selected, sweep, order

    def train_linear_classifier(
        self,
        latents,
        labels,
        val_size=0.2,
        batch_size=32,
        num_epochs=20,
        lr=1e-4,
        weight_decay=5e-2,
        use_l1=False,
        lambda_l1=1e-3,
        verbose=True,
        seed=None,
    ):
        if verbose:
            print(
                f"Training linear classifier"
                f"{' with L1 regularization' if use_l1 else ''}..."
            )
        if use_l1 and verbose:
            print(f"L1 lambda: {lambda_l1}")

        if val_size is not None and val_size > 0:
            train_latents, test_latents, train_labels, test_labels = train_test_split(
                latents,
                labels,
                test_size=val_size,
                random_state=42,
                stratify=labels,
            )
            if verbose:
                print(f"Using internal validation split with val_size={val_size}")
        else:
            train_latents = latents
            test_latents = latents
            train_labels = labels
            test_labels = labels
            if verbose:
                print(
                    "Skipping internal validation split; "
                    "reported metrics will be computed on the provided training data."
                )

        train_latents_tensor = torch.tensor(train_latents, dtype=torch.float32)
        test_latents_tensor = torch.tensor(test_latents, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

        train_ds = TensorDataset(train_latents_tensor, train_labels_tensor)
        test_ds = TensorDataset(test_latents_tensor, test_labels_tensor)
        loader_kwargs = dataloader_kwargs()
        loader_generator = None
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            loader_generator = torch.Generator()
            loader_generator.manual_seed(seed)
        train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
            **loader_kwargs,
        )
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, **loader_kwargs)

        input_dim = latents.shape[1]
        linear_classifier = nn.Linear(input_dim, 2).to(self.device)
        classifier_forward = maybe_compile(linear_classifier, name="linear_classifier")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            linear_classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        for epoch in range(num_epochs):
            linear_classifier.train()
            total_loss = torch.zeros((), device=self.device)
            for batch_latents, batch_labels in train_dataloader:
                batch_latents = batch_latents.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                with autocast_context(self.device, self.autocast_dtype):
                    outputs = classifier_forward(batch_latents)
                ce_loss = criterion(outputs, batch_labels)
                if use_l1:
                    l1_loss = lambda_l1 * torch.sum(torch.abs(linear_classifier.weight))
                    loss = ce_loss + l1_loss
                else:
                    loss = ce_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss = total_loss + loss.detach()

            avg_loss = float((total_loss / len(train_dataloader)).item())
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                if use_l1:
                    sparsity = (
                        (linear_classifier.weight.abs() < 1e-3).float().mean().item() * 100
                    )
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, "
                        f"Sparsity: {sparsity:.1f}%"
                    )
                else:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        linear_classifier.eval()
        all_preds = []
        all_eval_labels = []
        with torch.no_grad():
            for batch_latents, batch_labels in test_dataloader:
                batch_latents = batch_latents.to(self.device, non_blocking=True)
                with autocast_context(self.device, self.autocast_dtype):
                    outputs = classifier_forward(batch_latents)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_eval_labels.extend(batch_labels.numpy())

        test_acc = 100 * accuracy_score(all_eval_labels, all_preds)
        test_f1 = 100 * f1_score(all_eval_labels, all_preds, average="weighted")
        if verbose:
            print(
                f"Linear classifier training completed, test accuracy: {test_acc:.2f}%, "
                f"F1: {test_f1:.2f}%"
            )
            print("\nClassification report:")
            print(classification_report(all_eval_labels, all_preds))
        model_info = {
            "input_dim": input_dim,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "target_layer": self.target_layer,
        }
        return linear_classifier, test_acc, test_f1, model_info

    def train_multiple_linear_classifiers(
        self,
        latents,
        labels,
        num_probes=50,
        subset_fraction=0.5,
        seed=42,
        val_size=0.0,
        batch_size=32,
        num_epochs=20,
        lr=1e-4,
        weight_decay=5e-2,
        use_l1=False,
        lambda_l1=1e-3,
    ):
        if num_probes <= 0:
            raise ValueError("num_probes must be positive")
        if not (0.0 < subset_fraction <= 1.0):
            raise ValueError("subset_fraction must be in (0, 1]")

        latents_np = np.asarray(latents, dtype=np.float32)
        labels_np = np.asarray(labels)
        if latents_np.shape[0] != labels_np.shape[0]:
            raise ValueError("latents and labels must have the same number of rows")

        class0_idx = np.where(labels_np == 0)[0]
        class1_idx = np.where(labels_np == 1)[0]
        if len(class0_idx) == 0 or len(class1_idx) == 0:
            raise ValueError("Both classes must be present for probe training")

        class0_take = max(1, int(np.ceil(len(class0_idx) * subset_fraction)))
        class1_take = max(1, int(np.ceil(len(class1_idx) * subset_fraction)))

        classifiers = []
        importance_probe_vectors = []
        direction_probe_vectors = []
        probe_stats = []

        print(
            f"Training {num_probes} linear probes on independently sampled subsets "
            f"(subset_fraction={subset_fraction:.3f})..."
        )

        for probe_id in range(num_probes):
            rng = np.random.default_rng(seed + probe_id)
            sampled0 = rng.choice(class0_idx, size=class0_take, replace=False)
            sampled1 = rng.choice(class1_idx, size=class1_take, replace=False)
            subset_idx = np.concatenate([sampled0, sampled1])
            rng.shuffle(subset_idx)

            subset_latents = latents_np[subset_idx]
            subset_labels = labels_np[subset_idx]

            classifier, probe_acc, probe_f1, model_info = self.train_linear_classifier(
                latents=subset_latents,
                labels=subset_labels,
                val_size=val_size,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
                weight_decay=weight_decay,
                use_l1=use_l1,
                lambda_l1=lambda_l1,
                verbose=False,
                seed=seed + probe_id,
            )

            importance_vector = self.extract_difference_vector(classifier)
            direction_vector = self.extract_concept_vector(classifier, class_idx=1)
            classifiers.append(classifier)
            importance_probe_vectors.append(importance_vector)
            direction_probe_vectors.append(direction_vector)
            probe_stats.append(
                {
                    "probe_id": int(probe_id),
                    "seed": int(seed + probe_id),
                    "subset_size": int(len(subset_idx)),
                    "test_accuracy": float(probe_acc),
                    "test_f1": float(probe_f1),
                    "model_info": model_info,
                }
            )

            if (probe_id + 1) % 10 == 0 or probe_id == num_probes - 1:
                print(
                    f"Completed probes: {probe_id + 1}/{num_probes} "
                    f"(latest probe_id={probe_id})"
                )

        return (
            classifiers,
            np.stack(importance_probe_vectors, axis=0),
            np.stack(direction_probe_vectors, axis=0),
            probe_stats,
        )

    def aggregate_probe_vectors(self, importance_probe_vectors, direction_probe_vectors=None):
        importance_vectors_np = np.asarray(importance_probe_vectors, dtype=np.float32)
        if importance_vectors_np.ndim != 2:
            raise ValueError("importance_probe_vectors must have shape [num_probes, feature_dim]")
        if direction_probe_vectors is None:
            direction_vectors_np = importance_vectors_np
        else:
            direction_vectors_np = np.asarray(direction_probe_vectors, dtype=np.float32)
            if direction_vectors_np.shape != importance_vectors_np.shape:
                raise ValueError(
                    "direction_probe_vectors must have the same shape as importance_probe_vectors"
                )

        importance = np.mean(np.abs(importance_vectors_np), axis=0)
        signed_vector = np.mean(direction_vectors_np, axis=0)
        norm = np.linalg.norm(signed_vector)
        if norm > 0:
            signed_vector = signed_vector / norm
        return importance, signed_vector

    def select_dims_by_threshold(self, selected_indices, importance, fixed_threshold):
        selected_indices_np = np.asarray(selected_indices)
        importance_np = np.asarray(importance)
        if selected_indices_np.shape[0] != importance_np.shape[0]:
            raise ValueError("selected_indices and importance must have same length")
        mask = importance_np > fixed_threshold
        return selected_indices_np[mask]

    def evaluate_linear_classifier(self, classifier, latents, labels, batch_size=32):
        """Evaluate a trained linear classifier on arbitrary latents."""
        latents_tensor = torch.tensor(latents, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        eval_ds = TensorDataset(latents_tensor, labels_tensor)
        eval_dataloader = DataLoader(eval_ds, batch_size=batch_size, **dataloader_kwargs())
        classifier_forward = maybe_compile(classifier, name="linear_classifier_eval")

        classifier.eval()
        predictions = []
        gold_labels = []
        with torch.no_grad():
            for batch_latents, batch_labels in eval_dataloader:
                batch_latents = batch_latents.to(self.device, non_blocking=True)
                with autocast_context(self.device, self.autocast_dtype):
                    outputs = classifier_forward(batch_latents)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                gold_labels.extend(batch_labels.numpy())

        acc = 100 * accuracy_score(gold_labels, predictions)
        f1 = 100 * f1_score(gold_labels, predictions, average="weighted")
        return acc, f1

    def extract_difference_vector(self, classifier):
        output_weight = classifier.weight.detach().cpu().numpy()
        difference_vector = output_weight[1] - output_weight[0]
        norm = np.linalg.norm(difference_vector)
        if norm > 0:
            difference_vector = difference_vector / norm
        return difference_vector

    def extract_concept_vector(self, classifier, class_idx=1):
        weights = classifier.weight.detach().cpu().numpy()
        concept_vector = weights[class_idx]
        norm = np.linalg.norm(concept_vector)
        if norm > 0:
            concept_vector = concept_vector / norm
        return concept_vector

    def extract_concept_vectors(self, text_dataset, output_dir="concept_vectors"):
        os.makedirs(output_dir, exist_ok=True)

        if (
            os.path.exists(os.path.join(output_dir, "latents.npy"))
            and os.path.exists(os.path.join(output_dir, "labels.npy"))
        ):
            print(f"Loading precomputed latent representations from {output_dir}")
            original_latents = np.load(os.path.join(output_dir, "latents.npy"))
            original_labels = np.load(os.path.join(output_dir, "labels.npy"))
        else:
            original_latents, original_labels = self.precompute_latents(
                text_dataset=text_dataset,
                batch_size=4,
            )
            self._save_precompute_latent_representations(
                output_dir,
                original_latents,
                original_labels,
            )

        latents_np, selected_indices = self.select_important_features(original_latents)
        feature_dim = latents_np.shape[1]
        del original_latents
        gc.collect()

        classifier, test_acc, test_f1, model_info = self.train_linear_classifier(
            latents=latents_np,
            labels=np.array(original_labels),
            val_size=0.2,
            batch_size=32,
            num_epochs=20,
            lr=1e-4,
            weight_decay=5e-2,
            use_l1=self.use_l1,
            lambda_l1=self.lambda_l1,
        )

        model_info.update(
            {
                "selected_indices": selected_indices.tolist(),
                "original_dim": self.d_sae,
                "reduced_dim": feature_dim,
            }
        )

        truthful_vector = self.extract_concept_vector(classifier, class_idx=1)
        false_vector = self.extract_concept_vector(classifier, class_idx=0)
        difference_vector = self.extract_difference_vector(classifier)

        self._save_basic_vectors(
            output_dir,
            truthful_vector,
            false_vector,
            difference_vector,
        )

        results = {
            "selected_indices": selected_indices,
            "reduced_dim": feature_dim,
            "original_dim": self.d_sae,
            "difference_vector": difference_vector,
        }

        self._save_results(output_dir, results, model_info, classifier)
        print(f"Concept vector extraction completed! All results saved to {output_dir}")
        return results, classifier

    def _save_precompute_latent_representations(self, output_dir, latents, labels):
        np.save(os.path.join(output_dir, "latents.npy"), latents)
        np.save(os.path.join(output_dir, "labels.npy"), labels)
        print(f"Precomputed latent representations saved to {output_dir}")

    def _save_basic_vectors(self, output_dir, truthful_vector, false_vector, difference_vector):
        np.save(os.path.join(output_dir, "truthful_vector.npy"), truthful_vector)
        np.save(os.path.join(output_dir, "false_vector.npy"), false_vector)
        np.save(os.path.join(output_dir, "difference_vector.npy"), difference_vector)

    def _save_results(self, output_dir, results, model_info, classifier):
        # simple_repo writes lightweight JSON/NPY artifacts only.
        serializable_results = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in results.items()
        }
        with open(os.path.join(output_dir, "concept_vectors_summary.json"), "w", encoding="utf-8") as f:
            import json

            json.dump({"results": serializable_results, "model_info": model_info}, f, indent=2)


# ---------------------------------------------------------------------------
# Standalone helper functions for concept-vector evaluation
# ---------------------------------------------------------------------------

def analyze_truthfulness_with_concept(
    text,
    concept_vector,
    selected_indices,
    sae,
    language_model,
    target_layer=20,
    normalize=True,
    mean=None,
    std=None,
):
    tokens = language_model.to_tokens(text)
    with torch.no_grad():
        _, cache = language_model.run_with_cache(tokens)
        token_residual = cache["resid_post", target_layer][0, -1, :]
        full_latent = (
            sae.encode(token_residual.unsqueeze(0))
            .squeeze(0)
            .to(torch.float32)
            .cpu()
            .numpy()
        )
    reduced_latent = full_latent[selected_indices]

    norm_latent = reduced_latent / np.linalg.norm(reduced_latent)
    norm_concept = concept_vector / np.linalg.norm(concept_vector)
    similarity = np.dot(norm_latent, norm_concept)
    return similarity


def evaluate_concept_vector(
    test_dataset,
    concept_vector,
    selected_indices,
    sae,
    language_model,
    target_layer,
    normalize=True,
    mean=None,
    std=None,
    output_dir=".",
):
    os.makedirs(output_dir, exist_ok=True)
    predictions = []
    true_labels = []
    scores = []
    print(f"Evaluating concept vector on {len(test_dataset)} test samples...")
    for i in tqdm(range(len(test_dataset))):
        sample = test_dataset[i]
        if "text" in sample:
            text = sample["text"]
        else:
            raise ValueError("Could not find text field in dataset")
        if "label" in sample:
            true_label = sample["label"]
        else:
            raise ValueError("Could not find label field in dataset")
        score = analyze_truthfulness_with_concept(
            text,
            concept_vector,
            selected_indices,
            sae,
            language_model,
            target_layer,
            normalize,
            mean,
            std,
        )
        scores.append(score)
        predicted_label = 1 if score > 0 else 0
        predictions.append(predicted_label)
        true_labels.append(true_label)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("Test set evaluation results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(true_labels, predictions))
    return accuracy, f1, conf_matrix


def test_concept_vector_difference(
    test_dataset,
    difference_vector,
    selected_indices,
    sae,
    language_model,
    target_layer,
):
    """Test concept vector separation on test set."""
    leftscore = []
    rightscore = []
    for sample in tqdm(test_dataset, desc="Testing separation"):
        score = analyze_truthfulness_with_concept(
            sample["text"],
            difference_vector,
            selected_indices,
            sae,
            language_model,
            target_layer,
            normalize=False,
        )
        if sample["label"] == 0:
            leftscore.append(score)
        else:
            rightscore.append(score)

    leftscore = np.array(leftscore)
    rightscore = np.array(rightscore)

    separation = rightscore.mean() - leftscore.mean()
    right_correct = (rightscore > 0).sum()
    left_correct = (leftscore < 0).sum()
    accuracy = (right_correct + left_correct) / (len(rightscore) + len(leftscore)) * 100

    print("\nTest Set Results:")
    print(f"  Left:  mean={leftscore.mean():.4f}, std={leftscore.std():.4f}")
    print(f"  Right: mean={rightscore.mean():.4f}, std={rightscore.std():.4f}")
    print(f"  Separation: {separation:.4f}")
    print(f"  Accuracy:   {accuracy:.2f}%")

    return leftscore, rightscore, separation, accuracy
