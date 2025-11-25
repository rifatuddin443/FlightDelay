"""Additional advanced techniques: Ensemble methods and data augmentation."""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class EnsembleThresholdOptimizer:
    """Train multiple models with different random seeds and combine predictions.
    
    Problem: Single model might be miscalibrated due to DP noise.
    Solution: Train 3-5 models, average logits, then optimize threshold.
    """
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
        self.optimal_threshold = 0.5
    
    def predict_ensemble(self, data_loader, device='cuda') -> Tuple[np.ndarray, np.ndarray]:
        """Get averaged predictions from all models."""
        all_logits = []
        all_labels = []
        
        for model in self.models:
            model.eval()
            model_logits = []
            labels = []
            
            with torch.no_grad():
                for batch in data_loader:
                    # Your forward pass here
                    logits, _ = model(batch)
                    model_logits.append(logits.cpu().numpy())
                    labels.append(batch.y_cls.cpu().numpy())
            
            all_logits.append(np.concatenate(model_logits))
            all_labels.append(np.concatenate(labels))
        
        # Average logits
        avg_logits = np.mean(all_logits, axis=0)
        labels = all_labels[0]  # All should be same
        
        return avg_logits, labels
    
    def tune_threshold(self, val_loader, device='cuda'):
        """Find optimal threshold on validation set."""
        logits, labels = self.predict_ensemble(val_loader, device)
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.linspace(0.01, 0.99, 200):
            preds = (probs >= threshold).astype(int)
            
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        print(f"✅ Ensemble optimal threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
        return best_threshold


def augment_minority_class_temporal(
    x_seqs: np.ndarray,
    y_reg_seqs: np.ndarray,
    y_cls_seqs: np.ndarray,
    augment_factor: int = 3,
    noise_std: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augment minority class samples with temporal jittering and noise.
    
    Problem: Too few on-time samples for model to learn.
    Solution: Create synthetic on-time samples via small perturbations.
    
    Args:
        augment_factor: Create N copies of each minority sample
        noise_std: Gaussian noise standard deviation (5% default)
    """
    # Find minority samples (on-time flights)
    minority_mask = (y_cls_seqs.mean(axis=1) < 0.5)
    minority_indices = np.where(minority_mask)[0]
    
    if len(minority_indices) == 0:
        print("⚠️  No minority samples found, skipping augmentation")
        return x_seqs, y_reg_seqs, y_cls_seqs
    
    print(f"🔄 Augmenting {len(minority_indices)} minority samples × {augment_factor}...")
    
    augmented_x = []
    augmented_y_reg = []
    augmented_y_cls = []
    
    for idx in minority_indices:
        for _ in range(augment_factor):
            # Add Gaussian noise
            x_aug = x_seqs[idx] + np.random.normal(0, noise_std, x_seqs[idx].shape)
            y_reg_aug = y_reg_seqs[idx] + np.random.normal(0, noise_std * 0.5, y_reg_seqs[idx].shape)
            
            augmented_x.append(x_aug)
            augmented_y_reg.append(y_reg_aug)
            augmented_y_cls.append(y_cls_seqs[idx])  # Keep same label
    
    # Combine with original
    x_combined = np.concatenate([x_seqs, np.array(augmented_x)], axis=0)
    y_reg_combined = np.concatenate([y_reg_seqs, np.array(augmented_y_reg)], axis=0)
    y_cls_combined = np.concatenate([y_cls_seqs, np.array(augmented_y_cls)], axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(x_combined))
    x_combined = x_combined[indices]
    y_reg_combined = y_reg_combined[indices]
    y_cls_combined = y_cls_combined[indices]
    
    new_ratio = y_cls_combined.mean()
    print(f"✅ After augmentation: {len(x_combined)} samples, {new_ratio:.2%} delayed")
    
    return x_combined, y_reg_combined, y_cls_combined


def apply_mixup_augmentation(
    x_batch: torch.Tensor,
    y_cls_batch: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation for implicit regularization.
    
    MixUp: Create synthetic samples by interpolating between pairs.
    
    Usage in training loop:
        x, y_a, y_b, lam = apply_mixup_augmentation(x, y_cls)
        logits, _ = model(x)
        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
    
    Args:
        alpha: Beta distribution parameter (0.2 standard)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x_batch.size(0)
    index = torch.randperm(batch_size, device=x_batch.device)
    
    mixed_x = lam * x_batch + (1 - lam) * x_batch[index]
    y_a = y_cls_batch
    y_b = y_cls_batch[index]
    
    return mixed_x, y_a, y_b, lam


class FocalLoss(nn.Module):
    """Focal Loss to down-weight easy examples and focus on hard ones.
    
    FL(p) = -α(1-p)^γ log(p)
    
    Better than BCEWithLogitsLoss for severe imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs [N, 1]
            targets: Binary labels [N, 1]
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        
        # p_t: probability of correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # α_t: class-dependent weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal term: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma
        
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss using effective number of samples.
    
    Paper: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    
    Better than simple reweighting for extreme imbalance.
    """
    
    def __init__(self, samples_per_class: List[int], beta: float = 0.9999, loss_type: str = 'focal'):
        super().__init__()
        self.beta = beta
        self.loss_type = loss_type
        
        # Effective number of samples: (1 - β^n) / (1 - β)
        effective_num = [1.0 - beta ** n for n in samples_per_class]
        weights = [(1.0 - beta) / en for en in effective_num]
        
        # Normalize
        weights = [w / sum(weights) * len(weights) for w in weights]
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, 1]
            targets: [N, 1] binary labels
        """
        if self.loss_type == 'focal':
            loss = self.loss_fn(logits, targets)
        else:
            loss = self.loss_fn(logits, targets)
        
        # Apply class weights
        weights = targets * self.class_weights[1] + (1 - targets) * self.class_weights[0]
        weights = weights.to(logits.device)
        
        return (loss * weights).mean()


def compute_calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE) to diagnose miscalibration.
    
    Well-calibrated model: If model predicts 70% confident, 70% should be correct.
    
    Args:
        logits: Raw model outputs
        labels: True binary labels
        n_bins: Number of confidence bins
    
    Returns:
        ece: Expected Calibration Error
        bin_accuracies: Accuracy per confidence bin
        bin_confidences: Average confidence per bin
    """
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        bin_count = in_bin.sum()
        
        if bin_count > 0:
            bin_acc = (preds[in_bin] == labels[in_bin]).mean()
            bin_conf = probs[in_bin].mean()
            
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_count)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    # ECE: weighted average of |accuracy - confidence| per bin
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / bin_counts.sum()
    
    print(f"\n📊 CALIBRATION METRICS:")
    print(f"   Expected Calibration Error (ECE): {ece:.4f}")
    print(f"   {'Bin':>3} | {'Confidence':>10} | {'Accuracy':>8} | {'Count':>6} | {'Gap':>6}")
    print(f"   {'-'*3}-|-{'-'*10}-|-{'-'*8}-|-{'-'*6}-|-{'-'*6}")
    for i in range(n_bins):
        if bin_counts[i] > 0:
            gap = abs(bin_accuracies[i] - bin_confidences[i])
            print(f"   {i+1:3d} | {bin_confidences[i]:10.4f} | {bin_accuracies[i]:8.4f} | {bin_counts[i]:6.0f} | {gap:6.4f}")
    
    return ece, bin_accuracies, bin_confidences


def temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    initial_temp: float = 1.5,
) -> float:
    """Find optimal temperature to calibrate model predictions.
    
    Calibrated logits = logits / T
    Higher T → softer predictions (more uncertain)
    Lower T → sharper predictions (more confident)
    
    Use this after training:
        optimal_T = temperature_scaling(val_logits, val_labels)
        calibrated_logits = test_logits / optimal_T
    
    Returns:
        optimal_temperature: T value that minimizes NLL on validation set
    """
    from scipy.optimize import minimize_scalar
    
    def nll_with_temp(T):
        """Negative log-likelihood with temperature."""
        scaled_logits = logits / T
        probs = 1 / (1 + np.exp(-scaled_logits))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        
        nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        return nll
    
    result = minimize_scalar(nll_with_temp, bounds=(0.1, 10.0), method='bounded')
    optimal_T = result.x
    
    print(f"\n🌡️  TEMPERATURE SCALING:")
    print(f"   Optimal temperature: {optimal_T:.4f}")
    print(f"   Before: NLL = {nll_with_temp(1.0):.4f}")
    print(f"   After:  NLL = {nll_with_temp(optimal_T):.4f}")
    
    return optimal_T


# Example usage summary
USAGE_SUMMARY = """
🎯 HOW TO USE THESE TECHNIQUES:

1️⃣  ENSEMBLE (train_ensemble.py):
   ```python
   models = [train_model(seed=i) for i in range(5)]
   ensemble = EnsembleThresholdOptimizer(models)
   optimal_threshold = ensemble.tune_threshold(val_loader)
   ```

2️⃣  DATA AUGMENTATION (in data loading):
   ```python
   x_train, y_reg_train, y_cls_train = augment_minority_class_temporal(
       x_train, y_reg_train, y_cls_train,
       augment_factor=5, noise_std=0.05
   )
   ```

3️⃣  MIXUP (in training loop):
   ```python
   for x, y_reg, y_cls in train_loader:
       x_mixed, y_a, y_b, lam = apply_mixup_augmentation(x, y_cls)
       logits, _ = model(x_mixed)
       loss = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
   ```

4️⃣  FOCAL LOSS (replace BCEWithLogitsLoss):
   ```python
   criterion_cls = FocalLoss(alpha=0.25, gamma=2.0)
   ```

5️⃣  CLASS-BALANCED LOSS:
   ```python
   n_ontime = (y_cls_train < 0.5).sum()
   n_delayed = (y_cls_train >= 0.5).sum()
   criterion_cls = ClassBalancedLoss([n_ontime, n_delayed], beta=0.9999)
   ```

6️⃣  CALIBRATION CHECK (after training):
   ```python
   ece, _, _ = compute_calibration_metrics(val_logits, val_labels)
   optimal_T = temperature_scaling(val_logits, val_labels)
   calibrated_logits = test_logits / optimal_T
   ```

🏆 RECOMMENDED PIPELINE:
   1. Data: Temporal sampling + Class balancing
   2. Augmentation: Minority class augmentation (3-5×)
   3. Loss: Focal Loss or Class-Balanced Loss
   4. Training: Add MixUp if model overfits
   5. Post-training: Temperature scaling
   6. Inference: Ensemble + tuned threshold
"""

if __name__ == "__main__":
    print(USAGE_SUMMARY)
