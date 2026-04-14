from __future__ import annotations

"""
Explainability utilities for the hybrid PlantViT + Swin model.

This file contains:
- Grad-CAM support
- intermediate activation extraction using TorchVision
- optional LIME wrapper

Preprocessing for LIME inputs is intentionally left generic and should be
adapted later when the dataset pipeline is defined.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchvision.models.feature_extraction import create_feature_extractor


@dataclass
class XAIConfig:
    """Configuration for explainability methods."""

    gradcam_target_layer_plantvit: str = "plantvit.stage5"
    gradcam_target_layer_swin: str = "swin.features"
    lime_num_samples: int = 1000
    lime_top_labels: int = 5


class ForwardHookStore:
    """Stores activations and gradients for Grad-CAM."""

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

        handle = module.register_forward_hook(forward_hook)
        self._handles.append(handle)

    def clear(self) -> None:
        self.activations = None
        self.gradients = None

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class GradCAM:
    """Generic Grad-CAM implementation for spatial feature maps."""

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
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.hooks.activations
        gradients = self.hooks.gradients

        if activations.ndim != 4:
            raise ValueError(
                f"Grad-CAM expects 4D spatial activations, got shape {tuple(activations.shape)}"
            )

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def resolve_module(model: nn.Module, dotted_name: str) -> nn.Module:
    """Resolves a module using a dotted path."""
    current: nn.Module = model
    for part in dotted_name.split("."):
        if not hasattr(current, part):
            raise AttributeError(f"Module '{type(current).__name__}' has no attribute '{part}'")
        current = getattr(current, part)
        if not isinstance(current, nn.Module):
            raise TypeError(f"Resolved attribute '{part}' is not an nn.Module")
    return current


def build_gradcam_tools(
    model: nn.Module,
    xai_cfg: XAIConfig,
) -> Dict[str, GradCAM]:
    """Builds Grad-CAM helpers for PlantViT and Swin."""
    plantvit_module = resolve_module(model, xai_cfg.gradcam_target_layer_plantvit)
    swin_module = resolve_module(model, xai_cfg.gradcam_target_layer_swin)

    return {
        "plantvit_gradcam": GradCAM(model, plantvit_module),
        "swin_gradcam": GradCAM(model, swin_module),
    }


def build_torchvision_feature_extractors(model: nn.Module) -> Dict[str, nn.Module]:
    """Builds TorchVision-based intermediate feature extractors."""
    plantvit_nodes = {
        "plantvit.stage2": "plantvit_stage2",
        "plantvit.stage3": "plantvit_stage3",
        "plantvit.stage4": "plantvit_stage4",
        "plantvit.stage5": "plantvit_stage5",
    }

    swin_nodes = {
        "swin.features.1": "swin_stage1",
        "swin.features.3": "swin_stage2",
        "swin.features.5": "swin_stage3",
        "swin.features.7": "swin_stage4",
    }

    extractors = {
        "plantvit": create_feature_extractor(model, return_nodes=plantvit_nodes),
        "swin": create_feature_extractor(model, return_nodes=swin_nodes),
    }
    return extractors


class OptionalLIMEExplainer:
    """Optional image-level LIME wrapper."""

    def __init__(self, model: nn.Module, class_names: Sequence[str], device: str) -> None:
        self.model = model.eval()
        self.class_names = list(class_names)
        self.device = device

        try:
            from lime import lime_image  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("LIME is not installed. Install it with `pip install lime`.") from exc

        self._lime_image = lime_image
        self.explainer = self._lime_image.LimeImageExplainer()

    @torch.no_grad()
    def predict_proba_from_numpy(self, images_np):
        """Prediction wrapper expected by LIME.

        This assumes input as NHWC images. Real preprocessing will be added later.
        """
        if not isinstance(images_np, torch.Tensor):
            images = torch.from_numpy(images_np).float()
        else:
            images = images_np.float()

        if images.ndim != 4:
            raise ValueError("Expected LIME input as NHWC batch")

        images = images.permute(0, 3, 1, 2).contiguous().to(self.device)
        logits = self.model(images)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def explain(self, image_np, num_samples: int = 1000, top_labels: int = 5):
        """Generates a LIME explanation for a single image."""
        return self.explainer.explain_instance(
            image_np,
            classifier_fn=self.predict_proba_from_numpy,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
        )
