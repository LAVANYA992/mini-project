"""
land_classifier_infer.py

Lightweight inference wrapper around the trained MobileNetV3 land
classifier. Loaded once by the agent and reused across requests.

If no trained checkpoint exists yet (config.LAND_CLASSIFIER_MODEL_PATH),
this falls back to a clearly-flagged heuristic stub so the rest of the
pipeline (crop recommendation / yield / irrigation / agent) can still be
developed and demoed end-to-end before the DL model is trained.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

_TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image
except ImportError:
    _TORCH_AVAILABLE = False


class LandClassifier:
    """Wraps MobileNetV3 inference + EuroSAT-class -> land-condition mapping."""

    def __init__(self, model_path: str = config.LAND_CLASSIFIER_MODEL_PATH):
        self.model_path = model_path
        self.ready = False
        self.model = None
        self.class_names = config.EUROSAT_CLASSES
        self.device = None

        if _TORCH_AVAILABLE and os.path.exists(model_path):
            self._load_model()
        else:
            print(
                "[LandClassifier] No trained checkpoint found (or torch not installed). "
                "Falling back to heuristic stub — train models/train_land_classifier.py "
                "on EuroSAT to enable real inference."
            )

    def _load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.class_names = checkpoint["class_names"]

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(self.class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.ready = True
        print(f"[LandClassifier] Loaded trained model "
              f"(val_acc={checkpoint.get('best_val_acc', 'n/a')}) from {self.model_path}")

    def predict(self, image_path: str) -> dict:
        """
        Returns:
            {
                "eurosat_class": str,
                "land_condition": str,       # dry / wet / moist / vegetated / not_suitable
                "confidence": float,
                "suitable_for_agriculture": bool,
                "source": "model" | "heuristic_stub"
            }
        """
        if self.ready:
            return self._predict_with_model(image_path)
        return self._predict_stub(image_path)

    def _predict_with_model(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_idx = int(torch.argmax(probs).item())
            confidence = float(probs[top_idx].item())

        eurosat_class = self.class_names[top_idx]
        land_condition = config.LAND_CONDITION_MAP.get(eurosat_class, "unknown")
        suitable = land_condition != "not_suitable"

        return {
            "eurosat_class": eurosat_class,
            "land_condition": land_condition,
            "confidence": round(confidence, 4),
            "suitable_for_agriculture": suitable,
            "source": "model",
        }

    def _predict_stub(self, image_path: str) -> dict:
        """
        Deterministic, non-random placeholder used only when no trained
        checkpoint is available. Uses basic average-brightness/greenness
        heuristics on the raw pixels so the pipeline is still exercised
        meaningfully during development — NOT a substitute for the trained
        MobileNetV3 model.
        """
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(image_path).convert("RGB").resize((64, 64))
            arr = np.asarray(img).astype(float)
            r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
            brightness = (r + g + b) / 3

            if g > r and g > b:
                condition = "vegetated"
            elif b > r and b > g:
                condition = "wet"
            elif brightness > 180:
                condition = "dry"
            else:
                condition = "moist"

            return {
                "eurosat_class": "unknown (stub)",
                "land_condition": condition,
                "confidence": 0.5,
                "suitable_for_agriculture": condition != "not_suitable",
                "source": "heuristic_stub",
            }
        except Exception as e:
            return {
                "eurosat_class": "unknown",
                "land_condition": "unknown",
                "confidence": 0.0,
                "suitable_for_agriculture": True,
                "source": "heuristic_stub_error",
                "error": str(e),
            }


# Singleton instance reused by the agent / API
land_classifier = LandClassifier()
