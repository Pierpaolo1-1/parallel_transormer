from __future__ import annotations

"""
Main entry point for quick smoke tests and future project integration.

At this stage:
- builds the hybrid PlantViT + Swin model
- runs a simple forward pass on random input
- initializes XAI helpers

No dataset loading is performed yet.
"""

import torch

from models import build_hybrid_model
from explainability import XAIConfig, build_gradcam_tools, build_torchvision_feature_extractors


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_hybrid_model(num_classes=4).to(device)
    dummy = torch.randn(2, 3, 224, 224, device=device)

    logits = model(dummy)
    features = model.forward_features(dummy)

    print("Logits shape:", tuple(logits.shape))
    print("PlantViT embedding shape:", tuple(features["plantvit"].shape))
    print("Swin embedding shape:", tuple(features["swin"].shape))
    print("Fused embedding shape:", tuple(features["fused"].shape))

    xai_cfg = XAIConfig()
    gradcam_tools = build_gradcam_tools(model, xai_cfg)

    plantvit_cam = gradcam_tools["plantvit_gradcam"](dummy[:1])
    print("PlantViT Grad-CAM shape:", tuple(plantvit_cam.shape))

    extractors = build_torchvision_feature_extractors(model)
    activation_maps = extractors["plantvit"](dummy[:1])
    print("Extracted PlantViT nodes:", list(activation_maps.keys()))


if __name__ == "__main__":
    main()
