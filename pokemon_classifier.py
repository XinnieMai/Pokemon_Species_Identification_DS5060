"""
===============================================================================
Beyond the Pokédex: ViT and CNN Hybrid for Advanced Pokémon Identification
===============================================================================
DS5060 - University of Virginia, School of Data Science
Authors: Ryan Harrington, Christopher Lee, Xinnie Mai, Catherine Smith
 
This script implements the full experimentation pipeline described in the
project proposal, incorporating all feedback from the Milestone-1 review.
 
Key components:
  1. Dataset analysis & verification (long-tail distribution, resolution stats)
  2. Stratified 3-way split (70/15/15 train/val/test)
  3. Three backbone architectures: ResNet-50, Inception-ResNet-v2, ViT-B/16
  4. Four ablation studies:
       A. Backbone comparison
       B. Transfer learning vs random initialization
       C. Data augmentation on/off
       D. Per-generation accuracy analysis
  5. Grad-CAM interpretability visualizations
  6. Comprehensive metric reporting (top-1, top-5, macro F1)
  7. Visualization dashboard for all results
 
Usage:
    python pokemon_classifier.py --data_dir /path/to/pokemon_dataset \
                                  --output_dir ./results \
                                  --epochs 50 \
                                  --batch_size 32
===============================================================================
"""
 
import os
import sys
import json
import argparse
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, top_k_accuracy_score
)
 
from PIL import Image
 
# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
 
# ============================================================================
# SECTION 0: Configuration & Reproducibility
# ============================================================================
 
def set_seed(seed: int = 42):
    """
    Fix all random seeds for full reproducibility across runs.
    Covers Python, NumPy, PyTorch (CPU + CUDA), and CuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
 
def get_device():
    """Select best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
 
# ============================================================================
# SECTION 1: Dataset Analysis & Verification
# ============================================================================
# Feedback action: "Download the Kaggle dataset, report: total images,
# images per species (min/median/max/std), number of species with <10 images."
 
def analyze_dataset(data_dir: str) -> dict:
    """
    Perform comprehensive dataset verification as recommended in feedback.
    
    Reports:
      - Total image count and number of species
      - Per-species image distribution (min, median, max, std)
      - Long-tail statistics: species with <5, <10, <20 images
      - Image resolution distribution
      - Per-generation breakdown (maps species index to generation)
    
    Args:
        data_dir: Root directory of the Pokemon image dataset (ImageFolder format)
    
    Returns:
        Dictionary containing all analysis metrics and per-species counts
    """
    print("\n" + "=" * 70)
    print("  DATASET ANALYSIS & VERIFICATION")
    print("=" * 70)
 
    dataset = ImageFolder(data_dir)
    class_names = dataset.classes
    num_species = len(class_names)
 
    # Count images per species
    species_counts = Counter()
    for _, label in dataset.samples:
        species_counts[label] += 1
 
    counts = np.array(list(species_counts.values()))
 
    # ---------- Resolution sampling ----------
    # Sample up to 200 images to profile resolution distribution
    resolution_sample = random.sample(dataset.samples, min(200, len(dataset.samples)))
    widths, heights = [], []
    for img_path, _ in resolution_sample:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue
 
    # ---------- Generation mapping ----------
    # Pokémon species are conventionally grouped by National Pokédex number.
    # This mapping covers Gens 1–6 (721 species as referenced in the proposal).
    GENERATION_RANGES = {
        "Gen 1": (1, 151),
        "Gen 2": (152, 251),
        "Gen 3": (252, 386),
        "Gen 4": (387, 493),
        "Gen 5": (494, 649),
        "Gen 6": (650, 721),
    }
 
    stats = {
        "total_images": len(dataset),
        "num_species": num_species,
        "min_per_species": int(counts.min()),
        "max_per_species": int(counts.max()),
        "median_per_species": float(np.median(counts)),
        "mean_per_species": float(counts.mean()),
        "std_per_species": float(counts.std()),
        "species_lt_5": int((counts < 5).sum()),
        "species_lt_10": int((counts < 10).sum()),
        "species_lt_20": int((counts < 20).sum()),
        "resolution_width_mean": float(np.mean(widths)) if widths else 0,
        "resolution_height_mean": float(np.mean(heights)) if heights else 0,
        "class_names": class_names,
        "species_counts": dict(species_counts),
        "generation_ranges": GENERATION_RANGES,
    }
 
    # Pretty-print the analysis
    print(f"  Total images:          {stats['total_images']:,}")
    print(f"  Number of species:     {stats['num_species']}")
    print(f"  Images per species:")
    print(f"    Min:    {stats['min_per_species']}")
    print(f"    Median: {stats['median_per_species']:.0f}")
    print(f"    Max:    {stats['max_per_species']}")
    print(f"    Mean:   {stats['mean_per_species']:.1f}  (std={stats['std_per_species']:.1f})")
    print(f"  Long-tail species:")
    print(f"    < 5 images:  {stats['species_lt_5']} species")
    print(f"    < 10 images: {stats['species_lt_10']} species")
    print(f"    < 20 images: {stats['species_lt_20']} species")
    if widths:
        print(f"  Resolution (sampled {len(widths)} images):")
        print(f"    Mean width:  {stats['resolution_width_mean']:.0f} px")
        print(f"    Mean height: {stats['resolution_height_mean']:.0f} px")
    print("=" * 70)
 
    return stats


def analyze_dataset_from_object(dataset) -> dict:
    """
    Same as analyze_dataset but works on an already-loaded dataset object
    (ImageFolder or MergedImageFolder).
    """
    print("\n" + "=" * 70)
    print("  DATASET ANALYSIS & VERIFICATION")
    print("=" * 70)

    class_names = dataset.classes
    num_species = len(class_names)

    species_counts = Counter()
    for _, label in dataset.samples:
        species_counts[label] += 1

    counts = np.array(list(species_counts.values()))

    GENERATION_RANGES = {
        "Gen 1": (1, 151), "Gen 2": (152, 251), "Gen 3": (252, 386),
        "Gen 4": (387, 493), "Gen 5": (494, 649), "Gen 6": (650, 721),
    }

    stats = {
        "total_images": len(dataset),
        "num_species": num_species,
        "min_per_species": int(counts.min()),
        "max_per_species": int(counts.max()),
        "median_per_species": float(np.median(counts)),
        "mean_per_species": float(counts.mean()),
        "std_per_species": float(counts.std()),
        "species_lt_5": int((counts < 5).sum()),
        "species_lt_10": int((counts < 10).sum()),
        "species_lt_20": int((counts < 20).sum()),
        "class_names": class_names,
        "species_counts": dict(species_counts),
        "generation_ranges": GENERATION_RANGES,
    }

    print(f"  Total images:          {stats['total_images']:,}")
    print(f"  Number of species:     {stats['num_species']}")
    print(f"  Images per species:    min={stats['min_per_species']}, "
          f"median={stats['median_per_species']:.0f}, max={stats['max_per_species']}")
    print(f"  Long-tail: {stats['species_lt_5']} species <5 imgs, "
          f"{stats['species_lt_10']} <10 imgs")
    print("=" * 70)

    return stats
 
 
# ============================================================================
# SECTION 2: Stratified 3-Way Data Split
# ============================================================================
# Feedback action: "Create a proper 3-way split: 70/15/15 or 80/10/10
# train/val/test, stratified by species."
 
def create_stratified_splits(
    dataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Create stratified train/val/test splits with special handling for
    species that have very few images:
      - 1 image  → goes to train only
      - 2 images → train + val
      - 3+ images → stratified across all three splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    targets = np.array([label for _, label in dataset.samples])
    indices = np.arange(len(targets))

    # Count images per species
    species_counts = Counter(targets)

    # Separate indices by how many images the species has
    single_idx = []    # species with 1 image
    double_idx = []    # species with 2 images
    normal_idx = []    # species with 3+ images

    for idx in indices:
        count = species_counts[targets[idx]]
        if count == 1:
            single_idx.append(idx)
        elif count == 2:
            double_idx.append(idx)
        else:
            normal_idx.append(idx)

    # ---- Handle normal species (3+ images): full stratified split ----
    train_indices, val_indices, test_indices = [], [], []

    if normal_idx:
        normal_idx = np.array(normal_idx)
        normal_targets = targets[normal_idx]

        train_val_idx, test_idx = train_test_split(
            normal_idx, test_size=test_ratio,
            stratify=normal_targets, random_state=seed
        )
        relative_val = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=relative_val,
            stratify=targets[train_val_idx], random_state=seed
        )
        train_indices.extend(train_idx.tolist())
        val_indices.extend(val_idx.tolist())
        test_indices.extend(test_idx.tolist())

    # ---- Handle species with exactly 2 images: train + val ----
    rng = np.random.RandomState(seed)
    species_with_2 = [s for s, c in species_counts.items() if c == 2]
    for species in species_with_2:
        idxs = [i for i in double_idx if targets[i] == species]
        rng.shuffle(idxs)
        train_indices.append(idxs[0])
        val_indices.append(idxs[1])

    # ---- Handle species with exactly 1 image: train only ----
    train_indices.extend(single_idx)

    total = len(train_indices) + len(val_indices) + len(test_indices)
    print(f"\n  Data Split (stratified, with rare-species handling):")
    print(f"    Train: {len(train_indices):,} images ({100*len(train_indices)/total:.1f}%)")
    print(f"    Val:   {len(val_indices):,} images ({100*len(val_indices)/total:.1f}%)")
    print(f"    Test:  {len(test_indices):,} images ({100*len(test_indices)/total:.1f}%)")
    print(f"    Note:  {len(single_idx)} species with 1 image (train only)")
    print(f"           {len(species_with_2)} species with 2 images (train+val only)")

    return train_indices, val_indices, test_indices
 
 
# ============================================================================
# SECTION 3: Data Transforms & Augmentation
# ============================================================================
# Feedback action: augmentation ablation with specific transforms listed.
 
def get_transforms(model_name: str, augment: bool = True) -> dict:
    """
    Build train and evaluation transforms matched to each architecture's
    expected input resolution.
    
    Augmentation strategy (when enabled):
      - Random horizontal flip (p=0.5)
      - Random rotation (±15°)
      - Color jitter (brightness, contrast, saturation, hue)
      - Random resized crop
    
    The feedback specifically requested these augmentations for the
    augmentation ablation study.
    
    Args:
        model_name: One of "resnet50", "inception_resnet_v2", "vit_b_16"
        augment: Whether to apply data augmentation (True for training)
    
    Returns:
        Dict with 'train' and 'eval' transform pipelines
    """
    # Native input resolutions per architecture
    INPUT_SIZES = {
        "resnet50": 224,
        "inception_resnet_v2": 299,  # Native Inception resolution
        "vit_b_16": 224,
    }
    img_size = INPUT_SIZES.get(model_name, 224)
 
    # ImageNet normalization statistics (standard for pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
 
    # --- Evaluation transform: deterministic resize + center crop ---
    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),   # Slight oversize
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
 
    if augment:
        # --- Training transform: with geometric & color augmentations ---
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # --- No augmentation: resize only (for ablation study) ---
        train_transform = eval_transform
 
    return {"train": train_transform, "eval": eval_transform}
 
 
# ============================================================================
# SECTION 4: Model Architectures
# ============================================================================
# Implements all three backbones from the ablation plan:
#   - ResNet-50 (simple CNN baseline)
#   - Inception-ResNet-v2 (multi-scale CNN + residual, primary architecture)
#   - ViT-B/16 (Vision Transformer, attention-based)
 
def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Construct and return a classification model with the final layer
    replaced for the target number of classes.
    
    Supports the three architectures specified in the ablation plan.
    The `pretrained` flag controls the transfer-learning ablation:
      - True  → ImageNet-pretrained weights (transfer learning)
      - False → Random initialization (trains from scratch)
    
    Args:
        model_name: Architecture identifier
        num_classes: Number of output classes (721 for full Pokédex)
        pretrained: Whether to load ImageNet pretrained weights
        device: Target compute device
    
    Returns:
        PyTorch model moved to the specified device
    """
    print(f"  Building {model_name} (pretrained={pretrained}, classes={num_classes})")
 
    if model_name == "resnet50":
        # --- ResNet-50: Simple residual CNN baseline ---
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        # Replace the final fully-connected layer
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),          # Regularization for high-cardinality
            nn.Linear(in_features, num_classes)
        )
 
    elif model_name == "inception_resnet_v2":
        # --- Inception-ResNet-v2: Multi-scale CNN + Residual ---
        # Using timm library for the full Inception-ResNet-v2 implementation
        # Fallback to InceptionV3 from torchvision if timm is unavailable
        try:
            import timm
            model = timm.create_model(
                "inception_resnet_v2",
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=0.3,
            )
        except ImportError:
            print("    [FALLBACK] timm not found; using InceptionV3 from torchvision")
            weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.inception_v3(weights=weights, aux_logits=True)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )
            # Also replace the auxiliary classifier head
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
 
    elif model_name == "vit_b_16":
        # --- ViT-B/16: Vision Transformer with global self-attention ---
        # Feedback: "Commit to ViT-B/16 (pretrained on ImageNet-21K)"
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
 
    else:
        raise ValueError(f"Unknown model: {model_name}")
 
    return model.to(device)
 
 
# ============================================================================
# SECTION 5: Training Loop with Early Stopping
# ============================================================================
# Incorporates feedback: early stopping on val loss with patience=10,
# AdamW optimizer, cosine learning rate scheduler.
 
class EarlyStopping:
    """
    Monitors validation loss and stops training when no improvement
    is seen for `patience` consecutive epochs. Saves the best model
    weights for later restoration.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_weights = None
        self.triggered = False
 
    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False
 
    def restore_best(self, model: nn.Module):
        """Load the best weights back into the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
 
 
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """
    Execute one training epoch with optional mixed-precision (AMP).
    
    Returns:
        Tuple of (average_loss, accuracy) for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
 
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
 
        # Mixed-precision forward pass for faster training on GPU
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                # Handle InceptionV3 auxiliary outputs during training
                if isinstance(outputs, tuple):
                    outputs, aux_outputs = outputs
                    loss = criterion(outputs, labels) + 0.3 * criterion(aux_outputs, labels)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs, aux_outputs = outputs
                loss = criterion(outputs, labels) + 0.3 * criterion(aux_outputs, labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
 
    return running_loss / total, 100.0 * correct / total
 
 
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    """
    Evaluate the model on a dataset split.
    
    Returns a dictionary with:
      - loss, top1_acc, top5_acc, macro_f1
      - Per-class predictions for detailed analysis
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
 
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
 
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
 
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
 
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
 
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
 
    # --- Compute all metrics specified in the feedback ---
    top1 = 100.0 * accuracy_score(all_labels, all_preds)
    macro_f1 = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
 
    # Top-5 accuracy (feedback: "report top-5 alongside top-1")
    if num_classes >= 5:
        top5 = 100.0 * top_k_accuracy_score(
            all_labels, all_probs, k=5, labels=range(num_classes)
        )
    else:
        top5 = top1
 
    return {
        "loss": running_loss / len(all_labels),
        "top1_acc": top1,
        "top5_acc": top5,
        "macro_f1": macro_f1,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }
 
 
def run_experiment(
    model_name: str,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    pretrained: bool = True,
    device: torch.device = torch.device("cpu"),
    experiment_name: str = "experiment",
) -> dict:
    """
    Full training pipeline for one experiment configuration.
    
    Implements the feedback recommendations:
      - AdamW optimizer (lr=1e-4, weight_decay=1e-4)
      - Cosine annealing learning rate scheduler
      - Early stopping with patience=10 on validation loss
      - Mixed-precision training when CUDA is available
    
    Args:
        model_name: Architecture to train
        num_classes: Number of target classes
        train_loader, val_loader, test_loader: Data loaders for each split
        epochs: Maximum training epochs
        lr: Initial learning rate
        weight_decay: L2 regularization weight
        pretrained: Use ImageNet pretrained weights
        device: Compute device
        experiment_name: Label for logging
    
    Returns:
        Dict containing training history, test metrics, and model reference
    """
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
 
    model = build_model(model_name, num_classes, pretrained=pretrained, device=device)
 
    # --- Categorical cross-entropy (as specified in proposal) ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
 
    # --- AdamW optimizer (feedback recommendation over vanilla Adam) ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
 
    # --- Cosine annealing scheduler for smooth LR decay ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
 
    # --- Mixed-precision scaler for GPU acceleration ---
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
 
    early_stopping = EarlyStopping(patience=10)
 
    # History tracking for visualization
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_top5": [], "val_f1": [],
        "lr": [],
    }
 
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
 
        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
 
        # --- Validate ---
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
 
        # --- Record history ---
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["top1_acc"])
        history["val_top5"].append(val_metrics["top5_acc"])
        history["val_f1"].append(val_metrics["macro_f1"])
 
        # --- Log progress ---
        print(
            f"  Epoch {epoch:3d}/{epochs} │ "
            f"LR={current_lr:.2e} │ "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.1f}% │ "
            f"Val Loss={val_metrics['loss']:.4f} "
            f"Top1={val_metrics['top1_acc']:.1f}% "
            f"Top5={val_metrics['top5_acc']:.1f}% "
            f"F1={val_metrics['macro_f1']:.1f}%"
        )
 
        scheduler.step()
 
        # --- Early stopping check ---
        if early_stopping.step(val_metrics["loss"], model):
            print(f"  *** Early stopping triggered at epoch {epoch} ***")
            break
 
    # Restore best model weights
    early_stopping.restore_best(model)
 
    # --- Final test evaluation ---
    print(f"\n  Final Test Evaluation for: {experiment_name}")
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    print(f"    Test Top-1 Acc:  {test_metrics['top1_acc']:.2f}%")
    print(f"    Test Top-5 Acc:  {test_metrics['top5_acc']:.2f}%")
    print(f"    Test Macro F1:   {test_metrics['macro_f1']:.2f}%")
 
    return {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "pretrained": pretrained,
        "history": history,
        "test_metrics": test_metrics,
        "model": model,
    }
 
 
# ============================================================================
# SECTION 6: Grad-CAM Interpretability
# ============================================================================
# Feedback: "Grad-CAM visualizations may show the model attending to
# background colors rather than Pokémon features."
 
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates heatmaps showing which spatial regions of an input image
    contributed most to the model's classification decision.
    
    This is critical for determining whether the network focuses on
    biological features (ear shape, tail structure) or shortcuts
    (background color, artist signatures).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
 
        # Register hooks to capture activations and gradients
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
 
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
 
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
 
    def generate(self, input_tensor, class_idx=None):
        """
        Compute the Grad-CAM heatmap for a given input.
        
        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            class_idx: Target class (None = use predicted class)
        
        Returns:
            NumPy array heatmap normalized to [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
 
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
 
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
 
        # Global average pooling of gradients → channel importance weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # Only positive contributions
        cam = cam.squeeze().cpu().numpy()
 
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
 
 
def get_gradcam_target_layer(model, model_name):
    """
    Return the appropriate convolutional layer for Grad-CAM
    based on the model architecture.
    """
    if model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "inception_resnet_v2":
        try:
            return model.conv2d_7b  # timm Inception-ResNet-v2
        except AttributeError:
            return model.Mixed_7c    # torchvision InceptionV3 fallback
    elif model_name == "vit_b_16":
        # For ViT, use the last encoder block's layer norm
        return model.encoder.layers[-1].ln_1
    return None
 
 
# ============================================================================
# SECTION 7: Per-Generation & Per-Bin Analysis
# ============================================================================
# Feedback: "Report accuracy for Gen 1 (1–151), Gen 2 (152–251), etc."
# Also: "Report accuracy binned by training set size."
 
def per_generation_analysis(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    generation_ranges: dict,
) -> dict:
    """
    Compute accuracy broken down by Pokémon generation.
    
    Hypothesis from feedback: accuracy decreases for later generations
    because (a) fewer training images and (b) more complex/overlapping designs.
    
    Args:
        preds: Predicted labels from the test set
        labels: Ground-truth labels
        class_names: List of species names (indexed by label)
        generation_ranges: Dict mapping generation name → (start_idx, end_idx)
    
    Returns:
        Dict mapping generation name → accuracy percentage
    """
    gen_results = {}
 
    for gen_name, (start, end) in generation_ranges.items():
        # Select samples whose species index falls in this generation
        mask = (labels >= (start - 1)) & (labels < end)
        if mask.sum() == 0:
            continue
        gen_acc = 100.0 * accuracy_score(labels[mask], preds[mask])
        gen_f1 = 100.0 * f1_score(labels[mask], preds[mask], average="macro", zero_division=0)
        gen_results[gen_name] = {
            "accuracy": gen_acc,
            "macro_f1": gen_f1,
            "n_samples": int(mask.sum()),
        }
        print(f"    {gen_name}: Acc={gen_acc:.1f}%, F1={gen_f1:.1f}%, N={mask.sum()}")
 
    return gen_results
 
 
def accuracy_by_sample_bin(
    preds: np.ndarray,
    labels: np.ndarray,
    species_counts: dict,
    bins: list = None,
) -> dict:
    """
    Report accuracy binned by training set size per species.
    
    Feedback: "Report accuracy binned by training set size
    (species with 5–10 images, 10–20 images, 20+ images)."
    
    Args:
        preds, labels: Predictions and ground truth
        species_counts: Dict mapping class_idx → number of training images
        bins: List of (label, min_count, max_count) tuples
    
    Returns:
        Dict mapping bin label → accuracy
    """
    if bins is None:
        bins = [
            ("< 5 images", 0, 5),
            ("5–10 images", 5, 10),
            ("10–20 images", 10, 20),
            ("20+ images", 20, float("inf")),
        ]
 
    bin_results = {}
    for label, lo, hi in bins:
        # Identify species in this bin
        eligible = {
            cls for cls, count in species_counts.items()
            if lo <= count < hi
        }
        mask = np.isin(labels, list(eligible))
        if mask.sum() == 0:
            continue
        acc = 100.0 * accuracy_score(labels[mask], preds[mask])
        bin_results[label] = {"accuracy": acc, "n_samples": int(mask.sum())}
        print(f"    {label}: Acc={acc:.1f}% (N={mask.sum()})")
 
    return bin_results
 
 
# ============================================================================
# SECTION 8: Visualization Dashboard
# ============================================================================
 
def plot_training_curves(results_list: list, output_dir: str):
    """
    Generate training loss and accuracy curves for all experiments.
    Overlays experiments for direct comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training Curves: All Experiments", fontsize=16, fontweight="bold", y=0.98)
 
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_list)))
 
    for i, result in enumerate(results_list):
        name = result["experiment_name"]
        h = result["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        c = colors[i]
 
        # Training Loss
        axes[0, 0].plot(epochs, h["train_loss"], color=c, label=name, linewidth=2)
        axes[0, 0].set_title("Training Loss", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
 
        # Validation Loss
        axes[0, 1].plot(epochs, h["val_loss"], color=c, label=name, linewidth=2)
        axes[0, 1].set_title("Validation Loss", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
 
        # Validation Top-1 Accuracy
        axes[1, 0].plot(epochs, h["val_acc"], color=c, label=name, linewidth=2)
        axes[1, 0].set_title("Validation Top-1 Accuracy", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy (%)")
 
        # Validation Macro F1
        axes[1, 1].plot(epochs, h["val_f1"], color=c, label=name, linewidth=2)
        axes[1, 1].set_title("Validation Macro F1", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("F1 Score (%)")
 
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
 
 
def plot_backbone_comparison(results_list: list, output_dir: str):
    """
    Bar chart comparing Top-1, Top-5, and Macro F1 across backbones.
    Core visualization for the backbone ablation study.
    """
    backbone_results = [r for r in results_list if "backbone" in r["experiment_name"].lower()
                        or r["experiment_name"] in [
                            "ResNet-50 (Pretrained)",
                            "Inception-ResNet-v2 (Pretrained)",
                            "ViT-B/16 (Pretrained)"
                        ]]
    if not backbone_results:
        backbone_results = results_list[:3]  # fallback
 
    names = [r["experiment_name"] for r in backbone_results]
    top1 = [r["test_metrics"]["top1_acc"] for r in backbone_results]
    top5 = [r["test_metrics"]["top5_acc"] for r in backbone_results]
    f1s = [r["test_metrics"]["macro_f1"] for r in backbone_results]
 
    x = np.arange(len(names))
    width = 0.25
 
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, top1, width, label="Top-1 Acc", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x, top5, width, label="Top-5 Acc", color="#4CAF50", edgecolor="white")
    bars3 = ax.bar(x + width, f1s, width, label="Macro F1", color="#FF9800", edgecolor="white")
 
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
 
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Backbone Comparison: Test Set Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    path = os.path.join(output_dir, "backbone_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
 
 
def plot_ablation_summary(results_dict: dict, output_dir: str):
    """
    Comprehensive ablation summary showing the effect of each variable:
      - Transfer learning vs random init
      - Augmentation on vs off
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ablation Studies: Impact Analysis", fontsize=14, fontweight="bold")
 
    # --- Transfer Learning Ablation ---
    if "transfer" in results_dict:
        tl = results_dict["transfer"]
        categories = list(tl.keys())
        values = [tl[c]["top1_acc"] for c in categories]
        colors = ["#2196F3", "#F44336"]
        axes[0].bar(categories, values, color=colors, edgecolor="white", width=0.5)
        for i, v in enumerate(values):
            axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
        axes[0].set_title("Transfer Learning vs Random Init", fontweight="bold")
        axes[0].set_ylabel("Test Top-1 Accuracy (%)")
        axes[0].set_ylim(0, 105)
        axes[0].grid(axis="y", alpha=0.3)
 
    # --- Augmentation Ablation ---
    if "augmentation" in results_dict:
        aug = results_dict["augmentation"]
        categories = list(aug.keys())
        values = [aug[c]["top1_acc"] for c in categories]
        colors = ["#4CAF50", "#FF9800"]
        axes[1].bar(categories, values, color=colors, edgecolor="white", width=0.5)
        for i, v in enumerate(values):
            axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
        axes[1].set_title("Data Augmentation: On vs Off", fontweight="bold")
        axes[1].set_ylabel("Test Top-1 Accuracy (%)")
        axes[1].set_ylim(0, 105)
        axes[1].grid(axis="y", alpha=0.3)
 
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "ablation_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
 
 
def plot_generation_accuracy(gen_results: dict, output_dir: str):
    """
    Bar chart showing accuracy by Pokémon generation.
    Tests the hypothesis that later generations have lower accuracy.
    """
    if not gen_results:
        return None
 
    gens = list(gen_results.keys())
    accs = [gen_results[g]["accuracy"] for g in gens]
    f1s = [gen_results[g]["macro_f1"] for g in gens]
    n_samples = [gen_results[g]["n_samples"] for g in gens]
 
    fig, ax1 = plt.subplots(figsize=(10, 6))
 
    x = np.arange(len(gens))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, accs, width, label="Accuracy", color="#3F51B5", edgecolor="white")
    bars2 = ax1.bar(x + width / 2, f1s, width, label="Macro F1", color="#009688", edgecolor="white")
 
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Score (%)", fontsize=12)
    ax1.set_title("Per-Generation Performance Analysis", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(gens, fontsize=10)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)
 
    # Overlay sample counts as text
    for i, n in enumerate(n_samples):
        ax1.text(i, max(accs[i], f1s[i]) + 2, f"n={n}", ha="center", fontsize=9, color="gray")
 
    plt.tight_layout()
    path = os.path.join(output_dir, "generation_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
 
 
def plot_long_tail_distribution(species_counts: dict, output_dir: str):
    """
    Visualize the long-tail distribution of images per species.
    This directly addresses the feedback concern about data imbalance.
    """
    counts = sorted(species_counts.values(), reverse=True)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
    # Rank-ordered bar plot (long-tail shape)
    ax1.bar(range(len(counts)), counts, color="#7B1FA2", alpha=0.7, width=1.0)
    ax1.axhline(y=10, color="red", linestyle="--", alpha=0.7, label="10-image threshold")
    ax1.axhline(y=5, color="orange", linestyle="--", alpha=0.7, label="5-image threshold")
    ax1.set_xlabel("Species (ranked by count)", fontsize=11)
    ax1.set_ylabel("Number of Images", fontsize=11)
    ax1.set_title("Long-Tail Distribution", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
 
    # Histogram of counts
    ax2.hist(counts, bins=30, color="#00796B", edgecolor="white", alpha=0.8)
    ax2.axvline(x=10, color="red", linestyle="--", alpha=0.7, label="10 images")
    ax2.set_xlabel("Images per Species", fontsize=11)
    ax2.set_ylabel("Number of Species", fontsize=11)
    ax2.set_title("Distribution of Training Samples", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    path = os.path.join(output_dir, "long_tail_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
 
 
def plot_sample_bin_accuracy(bin_results: dict, output_dir: str):
    """
    Bar chart showing accuracy by training sample count bin.
    Tests whether rare species have systematically lower accuracy.
    """
    if not bin_results:
        return None
 
    bins = list(bin_results.keys())
    accs = [bin_results[b]["accuracy"] for b in bins]
 
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#F44336", "#FF9800", "#FFC107", "#4CAF50"][:len(bins)]
    bars = ax.bar(bins, accs, color=colors, edgecolor="white", width=0.5)
 
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
 
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy by Training Set Size (Per Species)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    path = os.path.join(output_dir, "sample_bin_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
 
 
def plot_confusion_matrix_top_n(preds, labels, class_names, n=20, output_dir="."):
    """
    Plot confusion matrix for the top-N most frequent species.
    Full 721-class matrix is unreadable, so we focus on the most common.
    """
    # Find top-N most frequent classes in the test set
    label_counts = Counter(labels)
    top_classes = [cls for cls, _ in label_counts.most_common(n)]
 
    mask = np.isin(labels, top_classes)
    filtered_preds = preds[mask]
    filtered_labels = labels[mask]
 
    # Remap to consecutive indices for the confusion matrix
    class_map = {c: i for i, c in enumerate(top_classes)}
    mapped_labels = np.array([class_map[l] for l in filtered_labels])
    mapped_preds = np.array([class_map.get(p, -1) for p in filtered_preds])
    valid = mapped_preds >= 0
    mapped_labels = mapped_labels[valid]
    mapped_preds = mapped_preds[valid]
 
    cm = confusion_matrix(mapped_labels, mapped_preds, labels=range(n))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
 
    top_names = [class_names[c][:12] for c in top_classes]  # Truncate long names
 
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=top_names, yticklabels=top_names, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Normalized Confusion Matrix (Top {n} Species)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
 
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix_top20.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
 
 
# ============================================================================
# SECTION 9: Main Experiment Orchestrator
# ============================================================================
 
def main():
    parser = argparse.ArgumentParser(
        description="Pokémon Classification: Full Ablation Study Pipeline"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to Pokémon dataset (ImageFolder format)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory for saving results and plots")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker threads")
    parser.add_argument(
        "--skip_ablations", nargs="*", default=[],
        choices=["backbone", "transfer", "augmentation", "generation"],
        help="Skip specific ablation studies to save time"
    )
    args = parser.parse_args()
 
    set_seed(args.seed)
    device = get_device()
    print(f"\n  Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
 
    # ---- Step 1: Dataset analysis ----
    full_dataset = PokemonSpriteDataset(args.data_dir)
    dataset_stats = analyze_dataset_from_object(full_dataset)
    num_classes = dataset_stats["num_species"]
 
    # ---- Step 2: Create 3-way stratified split ----
    full_dataset = PokemonSpriteDataset(args.data_dir)
    train_idx, val_idx, test_idx = create_stratified_splits(full_dataset, seed=args.seed)
 
    # ---- Step 3: Plot dataset distribution ----
    plot_long_tail_distribution(dataset_stats["species_counts"], args.output_dir)
 
    # ---- Collect all experiment results ----
    all_results = []
    ablation_dict = {}
 
    # ================================================================
    # ABLATION A: Backbone Comparison
    # ================================================================
    # Compare ResNet-50, Inception-ResNet-v2, ViT-B/16 — all pretrained
    backbones = [
        ("resnet50", "ResNet-50 (Pretrained)"),
        ("inception_resnet_v2", "Inception-ResNet-v2 (Pretrained)"),
        ("vit_b_16", "ViT-B/16 (Pretrained)"),
    ]
 
    if "backbone" not in args.skip_ablations:
        for model_name, exp_name in backbones:
            txf = get_transforms(model_name, augment=True)
 
            # Wrap subsets with appropriate transforms
            train_set = TransformSubset(full_dataset, train_idx, txf["train"])
            val_set = TransformSubset(full_dataset, val_idx, txf["eval"])
            test_set = TransformSubset(full_dataset, test_idx, txf["eval"])
 
            train_loader = DataLoader(
                train_set, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers,
                pin_memory=True, drop_last=True
            )
            val_loader = DataLoader(
                val_set, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers, pin_memory=True
            )
            test_loader = DataLoader(
                test_set, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers, pin_memory=True
            )
 
            result = run_experiment(
                model_name=model_name,
                num_classes=num_classes,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                lr=args.lr,
                pretrained=True,
                device=device,
                experiment_name=exp_name,
            )
            all_results.append(result)
 
    # ================================================================
    # ABLATION B: Transfer Learning vs Random Init
    # ================================================================
    # Hypothesis: transfer learning provides +15–25% accuracy improvement
    if "transfer" not in args.skip_ablations:
        model_name = "inception_resnet_v2"
        txf = get_transforms(model_name, augment=True)
        train_set = TransformSubset(full_dataset, train_idx, txf["train"])
        val_set = TransformSubset(full_dataset, val_idx, txf["eval"])
        test_set = TransformSubset(full_dataset, test_idx, txf["eval"])
 
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
 
        result_random = run_experiment(
            model_name=model_name,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            pretrained=False,  # <-- Random initialization
            device=device,
            experiment_name="Inception-ResNet-v2 (Random Init)",
        )
        all_results.append(result_random)
 
        # Pair with the pretrained version for comparison
        pretrained_result = next(
            (r for r in all_results if r["experiment_name"] == "Inception-ResNet-v2 (Pretrained)"),
            None
        )
        if pretrained_result:
            ablation_dict["transfer"] = {
                "Pretrained (ImageNet)": pretrained_result["test_metrics"],
                "Random Init": result_random["test_metrics"],
            }
 
    # ================================================================
    # ABLATION C: Data Augmentation On vs Off
    # ================================================================
    # Hypothesis: augmentation provides +3–5% accuracy overall
    if "augmentation" not in args.skip_ablations:
        model_name = "inception_resnet_v2"
 
        # --- Without augmentation ---
        txf_noaug = get_transforms(model_name, augment=False)
        train_set_noaug = TransformSubset(full_dataset, train_idx, txf_noaug["train"])
        val_set_noaug = TransformSubset(full_dataset, val_idx, txf_noaug["eval"])
        test_set_noaug = TransformSubset(full_dataset, test_idx, txf_noaug["eval"])
 
        train_loader = DataLoader(train_set_noaug, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set_noaug, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set_noaug, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
 
        result_noaug = run_experiment(
            model_name=model_name,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            pretrained=True,
            device=device,
            experiment_name="Inception-ResNet-v2 (No Augmentation)",
        )
        all_results.append(result_noaug)
 
        pretrained_result = next(
            (r for r in all_results if r["experiment_name"] == "Inception-ResNet-v2 (Pretrained)"),
            None
        )
        if pretrained_result:
            ablation_dict["augmentation"] = {
                "With Augmentation": pretrained_result["test_metrics"],
                "No Augmentation": result_noaug["test_metrics"],
            }
 
    # ================================================================
    # ABLATION D: Per-Generation Analysis
    # ================================================================
    if "generation" not in args.skip_ablations and all_results:
        best_result = max(all_results, key=lambda r: r["test_metrics"]["top1_acc"])
        print(f"\n  Per-Generation Analysis (using {best_result['experiment_name']})")
        gen_results = per_generation_analysis(
            best_result["test_metrics"]["preds"],
            best_result["test_metrics"]["labels"],
            dataset_stats["class_names"],
            dataset_stats["generation_ranges"],
        )
        plot_generation_accuracy(gen_results, args.output_dir)
 
        # Accuracy by training set size bin
        print(f"\n  Accuracy by Training Set Size Bin:")
        bin_results = accuracy_by_sample_bin(
            best_result["test_metrics"]["preds"],
            best_result["test_metrics"]["labels"],
            dataset_stats["species_counts"],
        )
        plot_sample_bin_accuracy(bin_results, args.output_dir)
 
    # ================================================================
    # Generate All Visualizations
    # ================================================================
    print(f"\n{'='*70}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
 
    if all_results:
        plot_training_curves(all_results, args.output_dir)
        plot_backbone_comparison(all_results, args.output_dir)
 
        if ablation_dict:
            plot_ablation_summary(ablation_dict, args.output_dir)
 
        # Confusion matrix for the best model
        best_result = max(all_results, key=lambda r: r["test_metrics"]["top1_acc"])
        plot_confusion_matrix_top_n(
            best_result["test_metrics"]["preds"],
            best_result["test_metrics"]["labels"],
            dataset_stats["class_names"],
            n=20,
            output_dir=args.output_dir,
        )
 
    # ---- Save experiment summary as JSON ----
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "total_images": dataset_stats["total_images"],
            "num_species": dataset_stats["num_species"],
            "long_tail_lt10": dataset_stats["species_lt_10"],
        },
        "experiments": [
            {
                "name": r["experiment_name"],
                "model": r["model_name"],
                "pretrained": r["pretrained"],
                "test_top1": r["test_metrics"]["top1_acc"],
                "test_top5": r["test_metrics"]["top5_acc"],
                "test_macro_f1": r["test_metrics"]["macro_f1"],
                "epochs_trained": len(r["history"]["train_loss"]),
            }
            for r in all_results
        ],
    }
    summary_path = os.path.join(args.output_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Experiment summary saved to: {summary_path}")
 
    print(f"\n{'='*70}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}\n")
 
 
# ============================================================================
# Helper: TransformSubset
# ============================================================================
# Applies different transforms to train/val/test subsets of the same dataset
 
class TransformSubset(Dataset):
    """
    Wraps a subset of an ImageFolder dataset with a specific transform.
    This allows the same underlying dataset to be used with different
    transforms for train (augmented) vs val/test (deterministic).
    """
    def __init__(self, dataset: ImageFolder, indices: list, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
 
    def __len__(self):
        return len(self.indices)
 
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.dataset.samples[real_idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class PokemonSpriteDataset(Dataset):
    """
    Loads Pokémon images from a nested folder structure where:
      - Folders represent art styles (default-front, official-artwork, gen-v, etc.)
      - Folders may contain subfolders of arbitrary depth
      - Filenames follow the pattern: 0001_Bulbasaur.png
      - Species identity is extracted from the filename, not the folder
    
    This unifies all art styles into a single dataset grouped by species,
    which is exactly what we want — the model must learn to recognize
    a Pokémon regardless of art style.
    """
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.samples = []       # List of (filepath, label_index)
        self.classes = []       # Sorted list of species names
        self.class_to_idx = {}  # species_name → integer label

        VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        # ---- First pass: walk all files, extract species names ----
        file_species_pairs = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VALID_EXTENSIONS:
                    continue

                # Parse species from filename like "0001_Bulbasaur.png"
                name_part = os.path.splitext(fname)[0]  # "0001_Bulbasaur"

                # Split on the first underscore to separate number from name
                parts = name_part.split("_", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    pokedex_num = parts[0]       # "0001"
                    species_name = parts[1]      # "Bulbasaur"
                    # Use "0001_Bulbasaur" as the class key so species
                    # with the same name but different numbers stay separate
                    class_key = f"{pokedex_num}_{species_name}"
                else:
                    # Fallback: use the whole filename as the class key
                    class_key = name_part

                filepath = os.path.join(dirpath, fname)
                file_species_pairs.append((filepath, class_key))

        # ---- Build unified class mapping ----
        unique_classes = sorted(set(pair[1] for pair in file_species_pairs))
        self.classes = unique_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

        # ---- Second pass: assign integer labels ----
        for filepath, class_key in file_species_pairs:
            label = self.class_to_idx[class_key]
            self.samples.append((filepath, label))

        # ---- Report what we found ----
        species_counts = Counter(label for _, label in self.samples)
        counts = np.array(list(species_counts.values()))

        print(f"\n  PokemonSpriteDataset loaded:")
        print(f"    Root:       {root_dir}")
        print(f"    Images:     {len(self.samples):,}")
        print(f"    Species:    {len(self.classes)}")
        print(f"    Per species: min={counts.min()}, median={np.median(counts):.0f}, "
              f"max={counts.max()}, mean={counts.mean():.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
 
# ============================================================================
# Entry point
# ============================================================================
 
if __name__ == "__main__":
    main()