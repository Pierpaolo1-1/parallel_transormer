from __future__ import annotations

"""
Training utilities for the hybrid PlantViT + Swin model.

This file contains:
- training configuration
- optimizer / loss / scheduler builders
- train / validation loop utilities
- optional K-fold loop helper

Dataset creation and preprocessing are intentionally excluded.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TrainingConfig:
    """Training configuration inspired by the paper."""

    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    eta_min: float = 1e-6
    num_folds: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EpochMetrics:
    """Simple container for epoch statistics."""

    loss: float
    accuracy: float


class TrainerFactory:
    """Builds criterion, optimizer, and scheduler."""

    @staticmethod
    def build_loss(cfg: TrainingConfig) -> nn.Module:
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    @staticmethod
    def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    @staticmethod
    def build_scheduler(
        optimizer: torch.optim.Optimizer,
        cfg: TrainingConfig,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.eta_min,
        )


def compute_top1_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Computes top-1 accuracy for a batch."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct) / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> EpochMetrics:
    """Runs one training epoch."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += compute_top1_accuracy(logits.detach(), labels)
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Empty training loader received.")

    return EpochMetrics(
        loss=running_loss / num_batches,
        accuracy=running_acc / num_batches,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[Tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: str,
) -> EpochMetrics:
    """Runs one validation epoch."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        running_acc += compute_top1_accuracy(logits, labels)
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Empty validation loader received.")

    return EpochMetrics(
        loss=running_loss / num_batches,
        accuracy=running_acc / num_batches,
    )


def run_training_loop(
    model: nn.Module,
    train_loader: Iterable[Tuple[Tensor, Tensor]],
    val_loader: Iterable[Tuple[Tensor, Tensor]],
    cfg: TrainingConfig,
) -> Dict[str, List[float]]:
    """Minimal complete training loop."""
    model = model.to(cfg.device)
    criterion = TrainerFactory.build_loss(cfg)
    optimizer = TrainerFactory.build_optimizer(model, cfg)
    scheduler = TrainerFactory.build_scheduler(optimizer, cfg)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -float("inf")
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, cfg.device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_metrics.loss)
        history["train_acc"].append(train_metrics.accuracy)
        history["val_loss"].append(val_metrics.loss)
        history["val_acc"].append(val_metrics.accuracy)
        history["lr"].append(current_lr)

        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch [{epoch + 1}/{cfg.epochs}] | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} | "
            f"lr={current_lr:.6e}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def run_kfold_training(
    model_builder,
    fold_loaders: Sequence[Tuple[Iterable[Tuple[Tensor, Tensor]], Iterable[Tuple[Tensor, Tensor]]]],
    cfg: TrainingConfig,
) -> List[Dict[str, List[float]]]:
    """Optional helper for K-fold training.

    `fold_loaders` must contain tuples of (train_loader, val_loader) for each fold.
    """
    histories: List[Dict[str, List[float]]] = []

    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders, start=1):
        print(f"\n{'=' * 70}")
        print(f"Starting fold {fold_idx}/{len(fold_loaders)}")
        print(f"{'=' * 70}")

        model = model_builder()
        history = run_training_loop(model, train_loader, val_loader, cfg)
        histories.append(history)

    return histories
