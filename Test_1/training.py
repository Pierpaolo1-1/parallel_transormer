
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TrainingConfig:
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    eta_min: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


class TrainerFactory:
    @staticmethod
    def build_loss(cfg: TrainingConfig) -> nn.Module:
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    @staticmethod
    def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    @staticmethod
    def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainingConfig):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)


def compute_top1_accuracy(logits: Tensor, targets: Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).sum().item()) / max(targets.numel(), 1)


def train_one_epoch(model: nn.Module, loader: Iterable[Tuple[Tensor, Tensor]], criterion: nn.Module, optimizer, device: str) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_top1_accuracy(logits.detach(), labels)
        n += 1

    return EpochMetrics(total_loss / max(n, 1), total_acc / max(n, 1))


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: Iterable[Tuple[Tensor, Tensor]], criterion: nn.Module, device: str) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc += compute_top1_accuracy(logits, labels)
        n += 1

    return EpochMetrics(total_loss / max(n, 1), total_acc / max(n, 1))


def run_training_loop(model: nn.Module, train_loader, val_loader, cfg: TrainingConfig) -> Dict[str, List[float]]:
    model = model.to(cfg.device)
    criterion = TrainerFactory.build_loss(cfg)
    optimizer = TrainerFactory.build_optimizer(model, cfg)
    scheduler = TrainerFactory.build_scheduler(optimizer, cfg)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, cfg.device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_metrics.loss)
        history["train_acc"].append(train_metrics.accuracy)
        history["val_loss"].append(val_metrics.loss)
        history["val_acc"].append(val_metrics.accuracy)
        history["lr"].append(lr)

        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch [{epoch+1}/{cfg.epochs}] | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} | "
            f"lr={lr:.6e}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
