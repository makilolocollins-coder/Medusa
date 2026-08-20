# ============================================================
# MEDUSA AI
# MammoSense V2 Model Engine
# ============================================================

import json

import torch
import torch.nn as nn
import timm

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO = "Makky07/MammoSense-breast-ultrasound"

MODEL_FILE = "mammosense_v2.pt"
CONFIG_FILE = "mammosense_v2_config.json"

DEFAULT_ARCHITECTURE = "vit_small_patch16_224"

DEFAULT_CLASSES = [
    "Normal",
    "Benign",
    "Malignant",
]

DEFAULT_IMAGE_SIZE = 224


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(filename):
    return hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="model",
    )


# ============================================================
# CONFIG
# ============================================================

def load_config():

    try:
        path = download_file(CONFIG_FILE)

        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return config

    except Exception:
        return {
            "architecture": DEFAULT_ARCHITECTURE,
            "classes": DEFAULT_CLASSES,
            "image_size": DEFAULT_IMAGE_SIZE,
        }


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(path):

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def get_state_dict(checkpoint):

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]

    else:
        state = checkpoint

    cleaned = {}

    for key, value in state.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    return cleaned


# ============================================================
# DETERMINE CLASSIFIER STRUCTURE
# ============================================================

def get_head_dimensions(state_dict):

    layers = []

    for key, tensor in state_dict.items():

        if (
            key.startswith("head.")
            and key.endswith(".weight")
            and torch.is_tensor(tensor)
            and tensor.ndim == 2
        ):

            number = int(key.split(".")[1])

            layers.append(
                (
                    number,
                    tensor.shape[1],
                    tensor.shape[0],
                )
            )

    layers.sort()

    if not layers:
        raise RuntimeError(
            "No classification head was found "
            "in the MammoSense checkpoint."
        )

    dimensions = []

    previous_output = None

    for _, input_features, output_features in layers:

        if (
            previous_output is not None
            and input_features != previous_output
        ):
            raise RuntimeError(
                "MammoSense classifier dimensions "
                "are inconsistent."
            )

        dimensions.append(output_features)
        previous_output = output_features

    return dimensions


# ============================================================
# MODEL
# ============================================================

class MammoSenseV2(nn.Module):

    def __init__(
        self,
        architecture,
        head_dimensions,
    ):

        super().__init__()

        self.backbone = timm.create_model(
            architecture,
            pretrained=False,
            num_classes=0,
        )

        input_features = self.backbone.num_features

        layers = []

        for i, output_features in enumerate(
            head_dimensions
        ):

            layers.append(
                nn.Linear(
                    input_features,
                    output_features,
                )
            )

            if i < len(head_dimensions) - 1:

                layers.append(
                    nn.ReLU()
                )

                layers.append(
                    nn.Dropout(0.0)
                )

            input_features = output_features

        self.head = nn.Sequential(*layers)

    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    state_dict,
    architecture,
):

    head_dimensions = get_head_dimensions(
        state_dict
    )

    model = MammoSenseV2(
        architecture=architecture,
        head_dimensions=head_dimensions,
    )

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "MammoSense checkpoint does not match "
            "the reconstructed model.\n\n"
            f"Head dimensions: {head_dimensions}\n\n"
            f"{error}"
        )

    return model


# ============================================================
# PREPROCESSING
# ============================================================

def create_transform(image_size):

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


# ============================================================
# MAMMOSENSE ENGINE
# ============================================================

class MammoSense:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.config = load_config()

        self.classes = self.config.get(
            "classes",
            DEFAULT_CLASSES,
        )

        self.architecture = self.config.get(
            "architecture",
            DEFAULT_ARCHITECTURE,
        )

        self.image_size = int(
            self.config.get(
                "image_size",
                DEFAULT_IMAGE_SIZE,
            )
        )

        model_path = download_file(
            MODEL_FILE
        )

        checkpoint = load_checkpoint(
            model_path
        )

        state_dict = get_state_dict(
            checkpoint
        )

        self.model = build_model(
            state_dict,
            self.architecture,
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        self.transform = create_transform(
            self.image_size
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    @torch.inference_mode()
    def predict(self, image):

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "MammoSense expects a PIL Image."
            )

        image = image.convert("RGB")

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        logits = self.model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        index = int(
            torch.argmax(probabilities)
        )

        prediction = self.classes[index]

        confidence = float(
            probabilities[index]
        )

        probability_dict = {
            name: float(probabilities[i])
            for i, name in enumerate(
                self.classes
            )
        }

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probability_dict,
            "architecture": self.architecture,
            "model": "MammoSense V2",
            "device": str(self.device),
        }


# ============================================================
# SINGLETON
# ============================================================

_ENGINE = None


def get_mammosense():

    global _ENGINE

    if _ENGINE is None:
        _ENGINE = MammoSense()

    return _ENGINE
