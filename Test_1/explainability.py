
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "XAIConfig",
    "GradCAM",
    "ForwardHookStore",
    "build_gradcam_tools",
    "ActivationExtractor",
    "build_activation_extractors",
    "save_cam_overlay",
    "save_activation_map",
    "OptionalLIMEExplainer",
    "resolve_module",
    "tensor_to_image_np",
]


@dataclass
class XAIConfig:
    gradcam_target_layer_plantvit: str = "plantvit.stage5"
    gradcam_target_layer_swin: str = "swin.stage3"
    lime_num_samples: int = 200
    lime_top_labels: int = 3


class ForwardHookStore:
    def __init__(self) -> None:
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, module: nn.Module) -> None:
        def forward_hook(_module: nn.Module, _inputs: Tuple[Tensor, ...], output: Tensor) -> None:
            self.activations = output

            def grad_hook(grad: Tensor) -> None:
                self.gradients = grad

            if isinstance(output, Tensor) and output.requires_grad:
                output.register_hook(grad_hook)

        self._handles.append(module.register_forward_hook(forward_hook))

    def clear(self) -> None:
        self.activations = None
        self.gradients = None


class GradCAM:
    def __init__(self, model: nn.Module, target_module: nn.Module) -> None:
        self.model = model
        self.hooks = ForwardHookStore()
        self.hooks.register(target_module)

    def __call__(self, x: Tensor, class_idx: Optional[int] = None) -> Tensor:
        self.model.eval()
        self.hooks.clear()

        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self.hooks.activations is None or self.hooks.gradients is None:
            raise RuntimeError("Grad-CAM could not capture activations or gradients.")

        activations = self.hooks.activations
        gradients = self.hooks.gradients

        if activations.ndim == 4 and activations.shape[1] < activations.shape[-1]:
            activations = activations.permute(0, 3, 1, 2).contiguous()
            gradients = gradients.permute(0, 3, 1, 2).contiguous()

        if activations.ndim != 4:
            raise ValueError(f"Expected 4D activations, got {tuple(activations.shape)}")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def resolve_module(model: nn.Module, dotted_name: str) -> nn.Module:
    current = model
    for part in dotted_name.split("."):
        current = getattr(current, part)
    return current


def build_gradcam_tools(model: nn.Module, cfg: XAIConfig) -> Dict[str, GradCAM]:
    return {
        "plantvit_gradcam": GradCAM(model, resolve_module(model, cfg.gradcam_target_layer_plantvit)),
        "swin_gradcam": GradCAM(model, resolve_module(model, cfg.gradcam_target_layer_swin)),
    }


class ActivationExtractor:
    def __init__(self, model: nn.Module, nodes: Dict[str, str]) -> None:
        self.outputs: Dict[str, Tensor] = {}
        self.handles = []

        for path, alias in nodes.items():
            module = resolve_module(model, path)

            def make_hook(name):
                def hook(_module, _inputs, output):
                    self.outputs[name] = output.detach()
                return hook

            self.handles.append(module.register_forward_hook(make_hook(alias)))

        self.model = model

    def __call__(self, x: Tensor) -> Dict[str, Tensor]:
        self.outputs = {}
        _ = self.model(x)
        return self.outputs


def build_activation_extractors(model: nn.Module) -> Dict[str, ActivationExtractor]:
    plantvit_nodes = {
        "plantvit.stage2": "plantvit_stage2",
        "plantvit.stage3": "plantvit_stage3",
        "plantvit.stage4": "plantvit_stage4",
        "plantvit.stage5": "plantvit_stage5",
    }
    swin_nodes = {
        "swin.stage1": "swin_stage1",
        "swin.stage2": "swin_stage2",
        "swin.stage3": "swin_stage3",
    }
    return {
        "plantvit": ActivationExtractor(model, plantvit_nodes),
        "swin": ActivationExtractor(model, swin_nodes),
    }


def tensor_to_image_np(x: Tensor) -> np.ndarray:
    tensor_cpu = x.detach().cpu().squeeze(0)
    image_np = tensor_cpu.permute(1, 2, 0).numpy()
    return np.clip(image_np, 0.0, 1.0)


def save_cam_overlay(image: Tensor, cam: Tensor, save_path: Path, title: str) -> None:
    image_np = tensor_to_image_np(image)
    cam_np = cam.detach().cpu().squeeze().numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(image_np)
    plt.imshow(cam_np, alpha=0.45)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def save_activation_map(feature_map: Tensor, save_path: Path, title: str) -> None:
    fmap = feature_map.detach().cpu()
    if fmap.ndim == 4 and fmap.shape[1] < fmap.shape[-1]:
        fmap = fmap.permute(0, 3, 1, 2).contiguous()
    fmap = fmap[0].mean(dim=0).numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(fmap)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


class OptionalLIMEExplainer:
    def __init__(self, model: nn.Module, class_names: Sequence[str], device: str) -> None:
        self.model = model.eval()
        self.class_names = list(class_names)
        self.device = device
        from lime import lime_image  # type: ignore
        from skimage.segmentation import mark_boundaries  # type: ignore
        self._mark_boundaries = mark_boundaries
        self.explainer = lime_image.LimeImageExplainer()

    @torch.no_grad()
    def predict_proba_from_numpy(self, images_np):
        if not isinstance(images_np, torch.Tensor):
            images = torch.from_numpy(images_np).float()
        else:
            images = images_np.float()

        images = images.permute(0, 3, 1, 2).contiguous().to(self.device)
        logits = self.model(images)
        return F.softmax(logits, dim=1).detach().cpu().numpy()

    def save_explanation(self, image: Tensor, save_path: Path, num_samples: int = 200, top_labels: int = 3) -> None:
        image_np = tensor_to_image_np(image)
        explanation = self.explainer.explain_instance(
            image_np,
            classifier_fn=self.predict_proba_from_numpy,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],  # type: ignore
            positive_only=True,
            num_features=8,
            hide_rest=False,
        )

        plt.figure(figsize=(4, 4))
        plt.imshow(self._mark_boundaries(temp / max(temp.max(), 1e-8), mask))
        plt.axis("off")
        plt.title("LIME explanation")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
