"""Checkpoint loading, image preprocessing, and command-line inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from par_model.model import MultiTaskPAR

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DISPLAY_LABELS = {
    "Gender": {"0": "Male", "1": "Female"},
    "Glasses": {"0": "No", "1": "Yes"},
}


def preprocess_image(image: Image.Image, height: int, width: int) -> torch.Tensor:
    """Resize and ImageNet-normalize one RGB person crop."""
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


class PersonAttributePredictor:
    """Load a project checkpoint and produce human-readable predictions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=self.device,
            weights_only=True,
        )
        self.target_cols = checkpoint["target_cols"]
        self.label_maps = checkpoint["label_maps"]
        self.inv_label_maps = checkpoint["inv_label_maps"]
        self.height = int(checkpoint.get("img_h", 256))
        self.width = int(checkpoint.get("img_w", 128))

        num_classes = {
            task: len(self.label_maps[task]) for task in self.target_cols
        }
        self.model = MultiTaskPAR(
            num_classes,
            model_name=checkpoint.get("model_name", "convnext_small"),
            pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: str | Path | Image.Image) -> dict[str, dict[str, Any]]:
        """Predict all attributes for one path or PIL image."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as pil_image:
                inputs = preprocess_image(
                    pil_image, self.height, self.width
                ).to(self.device)
        else:
            inputs = preprocess_image(image, self.height, self.width).to(self.device)
        outputs = self.model(inputs)

        predictions: dict[str, dict[str, Any]] = {}
        for task in self.target_cols:
            probabilities = torch.softmax(outputs[task], dim=1)[0]
            class_id = int(probabilities.argmax().item())
            raw_label = str(self.inv_label_maps[task][class_id])
            label = DISPLAY_LABELS.get(task, {}).get(raw_label, raw_label)
            predictions[task] = {
                "label": label,
                "raw_label": raw_label,
                "confidence": round(float(probabilities[class_id].item()), 6),
            }
        return predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict six pedestrian attributes from one person crop."
    )
    parser.add_argument("image", type=Path, help="Path to an input person crop")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/best_model.pth"),
        help="Path to the released PyTorch checkpoint",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        help="Inference device; defaults to CUDA when available, otherwise CPU",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = PersonAttributePredictor(args.checkpoint, device=args.device)
    result = {
        "image": str(args.image),
        "predictions": predictor.predict(args.image),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
