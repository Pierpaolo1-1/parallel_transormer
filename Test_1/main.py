
from __future__ import annotations

from pathlib import Path

import torch

from models import build_hybrid_model
from preprocessing import PreprocessingConfig, build_digits_dataloaders
from training import TrainingConfig, TrainerFactory, run_training_loop, validate_one_epoch
from explainability import (
    XAIConfig,
    build_gradcam_tools,
    build_activation_extractors,
    save_cam_overlay,
    save_activation_map,
    OptionalLIMEExplainer,
)


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    prep_cfg = PreprocessingConfig(image_size=64, batch_size=32)
    train_loader, val_loader, test_loader, class_names = build_digits_dataloaders(prep_cfg)

    train_cfg = TrainingConfig(epochs=2, batch_size=32, learning_rate=3e-4)
    model = build_hybrid_model(num_classes=len(class_names)).to(train_cfg.device)

    print("Starting training...")
    history = run_training_loop(model, train_loader, val_loader, train_cfg)

    criterion = TrainerFactory.build_loss(train_cfg)
    test_metrics = validate_one_epoch(model, test_loader, criterion, train_cfg.device)
    print(f"Test loss={test_metrics.loss:.4f} | Test acc={test_metrics.accuracy:.4f}")

    batch_images, batch_labels = next(iter(test_loader))
    image = batch_images[:1].to(train_cfg.device)
    label = int(batch_labels[0].item())

    model.eval()
    with torch.no_grad():
        logits = model(image)
        pred = int(torch.argmax(logits, dim=1).item())

    xai_cfg = XAIConfig()
    gradcam_tools = build_gradcam_tools(model, xai_cfg)
    activation_extractors = build_activation_extractors(model)

    plantvit_cam = gradcam_tools["plantvit_gradcam"](image)
    swin_cam = gradcam_tools["swin_gradcam"](image)

    save_cam_overlay(image.cpu(), plantvit_cam.cpu(), output_dir / "gradcam_plantvit.png", f"PlantViT Grad-CAM | true={label} pred={pred}")
    save_cam_overlay(image.cpu(), swin_cam.cpu(), output_dir / "gradcam_swin.png", f"Swin-like Grad-CAM | true={label} pred={pred}")

    plantvit_acts = activation_extractors["plantvit"](image)
    swin_acts = activation_extractors["swin"](image)

    for name, fmap in plantvit_acts.items():
        save_activation_map(fmap, output_dir / f"{name}.png", title=name)

    for name, fmap in swin_acts.items():
        save_activation_map(fmap, output_dir / f"{name}.png", title=name)

    lime_saved = False
    try:
        lime_explainer = OptionalLIMEExplainer(model, class_names, train_cfg.device)
        lime_explainer.save_explanation(
            image.cpu(),
            output_dir / "lime_explanation.png",
            num_samples=xai_cfg.lime_num_samples,
            top_labels=xai_cfg.lime_top_labels,
        )
        lime_saved = True
    except Exception as exc:
        print(f"LIME skipped: {exc}")

    print("\nSaved files in outputs/:")
    for path in sorted(output_dir.glob("*")):
        print("-", path.name)

    print("\nSummary:")
    print("history keys:", list(history.keys()))
    print("predicted class:", pred)
    print("true class:", label)
    print("LIME saved:", lime_saved)


if __name__ == "__main__":
    main()
