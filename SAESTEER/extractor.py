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


class LinearConceptExtractor:
    def __init__(self, sae, language_model, target_layer=None, device='cuda', use_l1=False, lambda_l1=1e-3):
        self.sae = sae
        self.language_model = language_model
        self.target_layer = target_layer if target_layer is not None else language_model.cfg.n_layers - 1
        self.device = device
        self.sae.to(device)
        self.d_sae = sae.cfg.d_sae
        self.use_l1 = use_l1
        self.lambda_l1 = lambda_l1
        print(f"Initializing LinearConceptExtractor, using model layer {self.target_layer+1}/{language_model.cfg.n_layers}")

    def precompute_latents(self, text_dataset, batch_size=16):
        print(f"Precomputing latent representations, total {len(text_dataset['text'])} samples...")
        all_latents = []
        all_labels = []
        for i in tqdm(range(0, len(text_dataset['text']), batch_size)):
            batch_texts = text_dataset['text'][i:i+batch_size]
            batch_labels = text_dataset['label'][i:i+batch_size]
            batch_latents = []
            for text in batch_texts:
                tokens = self.language_model.to_tokens(text)
                with torch.no_grad():
                    _, cache = self.language_model.run_with_cache(
                        tokens,
                        stop_at_layer=self.target_layer + 1,
                        names_filter=lambda name: name == f'blocks.{self.target_layer}.hook_resid_post',
                    )
                    token_residual = cache['resid_post', self.target_layer][0, -1, :]
                    latent = self.sae.encode(token_residual.unsqueeze(0)).squeeze(0).to(torch.float32)
                    batch_latents.append(latent.cpu())
                del cache
                torch.cuda.empty_cache()
            all_latents.extend(batch_latents)
            all_labels.extend(batch_labels)

        return all_latents, all_labels

    def select_important_features(self, latents):
        """Remove all-zero columns and keep only non-zero features."""
        print("Performing feature selection, removing all-zero columns...")
        if isinstance(latents[0], torch.Tensor):
            latents_np = np.array([l.cpu().numpy() for l in latents])
        else:
            latents_np = np.array(latents)

        n_features = latents_np.shape[1]
        # Find columns that are not all zeros
        non_zero_mask = ~np.all(latents_np == 0, axis=0)
        selected_indices = np.where(non_zero_mask)[0]
        selected_latents = latents_np[:, selected_indices]

        n_zero_cols = n_features - len(selected_indices)
        print(f"All-zero columns: {n_zero_cols}/{n_features} | Active columns: {len(selected_indices)}")
        print(f"Feature selection completed, reduced from {n_features} to {len(selected_indices)} features")
        return selected_latents, selected_indices


    def train_linear_classifier(self, latents, labels, val_size=0.2, batch_size=32,
                                num_epochs=20, lr=1e-4, weight_decay=5e-2, use_l1=False, lambda_l1=1e-3):
     
        print(f"Training linear classifier{' with L1 regularization' if use_l1 else ''}...")
        if use_l1:
            print(f"L1 lambda: {lambda_l1}")
        train_latents, test_latents, train_labels, test_labels = train_test_split(
            latents, labels, test_size=val_size, random_state=42, stratify=labels
        )
        train_latents_tensor = torch.tensor(train_latents, dtype=torch.float32)
        test_latents_tensor = torch.tensor(test_latents, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
        train_ds = TensorDataset(train_latents_tensor, train_labels_tensor)
        test_ds = TensorDataset(test_latents_tensor, test_labels_tensor)
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size)
        input_dim = latents.shape[1]
        linear_classifier = nn.Linear(input_dim, 2).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(linear_classifier.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            linear_classifier.train()
            total_loss = 0
            for batch_latents, batch_labels in train_dataloader:
                batch_latents = batch_latents.to(self.device)
                batch_labels = batch_labels.to(self.device)
                outputs = linear_classifier(batch_latents)
                ce_loss = criterion(outputs, batch_labels)
                # Add L1 regularization if enabled
                if use_l1:
                    l1_loss = lambda_l1 * torch.sum(torch.abs(linear_classifier.weight))
                    loss = ce_loss + l1_loss
                else:
                    loss = ce_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dataloader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                if use_l1:
                    sparsity = (linear_classifier.weight.abs() < 1e-3).float().mean().item() * 100
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Sparsity: {sparsity:.1f}%")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        linear_classifier.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_latents, batch_labels in test_dataloader:
                batch_latents = batch_latents.to(self.device)
                outputs = linear_classifier(batch_latents)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        test_acc = 100 * accuracy_score(all_labels, all_preds)
        test_f1 = 100 * f1_score(all_labels, all_preds, average='weighted')
        print(f"Linear classifier training completed, test accuracy: {test_acc:.2f}%, F1: {test_f1:.2f}%")
        print("\nClassification report:")
        print(classification_report(all_labels, all_preds))
        model_info = {
            'input_dim': input_dim,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'target_layer': self.target_layer
        }
        return linear_classifier, test_acc, test_f1, model_info

    def extract_difference_vector(self, classifier):
        output_weight = classifier.weight.detach().cpu().numpy()
        difference_vector = output_weight[1] - output_weight[0]
        difference_vector = difference_vector / np.linalg.norm(difference_vector)
        return difference_vector

    def extract_concept_vector(self, classifier, class_idx=1):
        weights = classifier.weight.detach().cpu().numpy()
        concept_vector = weights[class_idx]
        concept_vector = concept_vector / np.linalg.norm(concept_vector)
        return concept_vector

    def extract_concept_vectors(self, text_dataset, output_dir="concept_vectors"):
        os.makedirs(output_dir, exist_ok=True)
        # Precompute latent representations
        if os.path.exists(os.path.join(output_dir, "latents.npy")) and os.path.exists(os.path.join(output_dir, "labels.npy")):
            print(f"Loading precomputed latent representations from {output_dir}")
            original_latents = np.load(os.path.join(output_dir, "latents.npy"))
            original_labels = np.load(os.path.join(output_dir, "labels.npy"))
        else:
            original_latents, original_labels = self.precompute_latents(
                text_dataset=text_dataset, batch_size=4
            )
            self._save_precompute_latent_representations(output_dir, original_latents, original_labels)

        
       
        
        # Feature selection
        latents_np, selected_indices = self.select_important_features(original_latents)
        feature_dim = latents_np.shape[1]
        del original_latents
        gc.collect()
        


        # Train linear classifier
        classifier, test_acc, test_f1, model_info = self.train_linear_classifier(
            latents=latents_np,
            labels=np.array(original_labels),
            val_size=0.2, batch_size=32, num_epochs=20, lr=1e-4, weight_decay=5e-2, use_l1=self.use_l1, lambda_l1=self.lambda_l1
        )
        
        

        model_info.update({
            'selected_indices': selected_indices.tolist(),
            'original_dim': self.d_sae,
            'reduced_dim': feature_dim
        })
        
        # Extract basic concept vectors
        truthful_vector = self.extract_concept_vector(classifier, class_idx=1)
        false_vector = self.extract_concept_vector(classifier, class_idx=0)
        difference_vector = self.extract_difference_vector(classifier)

        self._save_basic_vectors(output_dir, truthful_vector, false_vector, difference_vector)

        results = {
            'selected_indices': selected_indices,
            'reduced_dim': feature_dim,
            'original_dim': self.d_sae,
            'difference_vector': difference_vector,
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
        torch.save(results, os.path.join(output_dir, "concept_vectors_full.pt"))
        torch.save(model_info, os.path.join(output_dir, "model_info.pt"))
        model_save_path = os.path.join(output_dir, f"linear_classifier_layer_{self.target_layer}.pt")
        torch.save({
            'model_state_dict': classifier.state_dict(),
            'model_info': model_info
        }, model_save_path)


# ---------------------------------------------------------------------------
# Standalone helper functions for concept-vector evaluation
# ---------------------------------------------------------------------------

def analyze_truthfulness_with_concept(text, concept_vector, selected_indices, sae, language_model,
                                      target_layer=20, normalize=True, mean=None, std=None):
    tokens = language_model.to_tokens(text)
    with torch.no_grad():
        logits, cache = language_model.run_with_cache(tokens)
        token_residual = cache['resid_post', target_layer][0, -1, :]
        full_latent = sae.encode(token_residual.unsqueeze(0)).squeeze(0).to(torch.float32).cpu().numpy()
    reduced_latent = full_latent[selected_indices]
   
    norm_latent = reduced_latent / np.linalg.norm(reduced_latent)
    norm_concept = concept_vector / np.linalg.norm(concept_vector)
    similarity = np.dot(norm_latent, norm_concept)
    return similarity




def evaluate_concept_vector(test_dataset, concept_vector, selected_indices, sae, language_model,
                            target_layer, normalize=True, mean=None, std=None, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    predictions = []
    true_labels = []
    scores = []
    print(f"Evaluating concept vector on {len(test_dataset)} test samples...")
    for i in tqdm(range(len(test_dataset))):
        sample = test_dataset[i]
        if 'text' in sample:
            text = sample['text']
        else:
            raise ValueError("Could not find text field in dataset")
        if 'label' in sample:
            true_label = sample['label']
        else:
            raise ValueError("Could not find label field in dataset")
        score = analyze_truthfulness_with_concept(
            text, concept_vector, selected_indices, sae, language_model,
            target_layer, normalize, mean, std
        )
        scores.append(score)
        predicted_label = 1 if score > 0 else 0
        predictions.append(predicted_label)
        true_labels.append(true_label)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)
    print(f"Test set evaluation results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(true_labels, predictions))
    return accuracy, f1, conf_matrix


def test_concept_vector_difference(test_dataset, difference_vector, selected_indices, sae, language_model, target_layer):
    """Test concept vector separation on test set. Returns scores, separation, and accuracy."""
    leftscore = []
    rightscore = []
    for sample in tqdm(test_dataset, desc="Testing separation"):
        score = analyze_truthfulness_with_concept(
            sample['text'], difference_vector, selected_indices, sae, language_model,
            target_layer, normalize=False
        )
        if sample['label'] == 0:
            leftscore.append(score)
        else:
            rightscore.append(score)

    leftscore = np.array(leftscore)
    rightscore = np.array(rightscore)

    # Calculate metrics
    separation = rightscore.mean() - leftscore.mean()
    right_correct = (rightscore > 0).sum()
    left_correct = (leftscore < 0).sum()
    accuracy = (right_correct + left_correct) / (len(rightscore) + len(leftscore)) * 100

    print(f"\nTest Set Results:")
    print(f"  Left:  mean={leftscore.mean():.4f}, std={leftscore.std():.4f}")
    print(f"  Right: mean={rightscore.mean():.4f}, std={rightscore.std():.4f}")
    print(f"  Separation: {separation:.4f}")
    print(f"  Accuracy:   {accuracy:.2f}%")

    return leftscore, rightscore, separation, accuracy