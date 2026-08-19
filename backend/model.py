import json
import os
from pathlib import Path

import numpy as np
import torch
import timm

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download


# ============================================================
# MEDUSA / MAMMOSENSE MODEL CONFIGURATION
# ============================================================

HF_REPO = "Makky07/MammoSense-breast-ultrasound"

# Current files in the Hugging Face repository
MODEL_FILENAME = "mammosense_v2.pt"
CONFIG_FILENAME = "mammosense_v2_config.json"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CONFIG
# ============================================================

print("=" * 60)
print("MEDUSA AI BACKEND")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Hugging Face repo: {HF_REPO}")


config_path = hf_hub_download(
    repo_id=HF_REPO,
    filename=CONFIG_FILENAME,
)

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)


CLASS_NAMES = config.get(
    "classes",
    ["Normal", "Benign", "Malignant"]
)

CLASS_TO_IDX = config.get(
    "class_to_idx",
    {
        "normal": 0,
        "benign": 1,
        "malignant": 2,
    }
)

print("Classes:", CLASS_NAMES)
print("Architecture:", config.get(
    "architecture",
    "vit_small_patch16_224"
))


# ============================================================
# DOWNLOAD MODEL FROM HUGGING FACE
# ============================================================

model_path = hf_hub_download(
    repo_id=HF_REPO,
    filename=MODEL_FILENAME,
)

print(f"Model downloaded/cached at:")
print(model_path)


# ============================================================
# CREATE MODEL
# ============================================================

model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=len(CLASS_NAMES),
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")


def load_checkpoint(path):
    """
    PyTorch 2.6 changed torch.load's default to
    weights_only=True.

    Our checkpoint is our own trusted model checkpoint,
    so we first try the safer weights-only mode and then
    fall back to weights_only=False if the checkpoint
    contains objects that the restricted unpickler cannot
    handle.
    """

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    except Exception as first_error:

        print(
            "weights_only=True could not load checkpoint."
        )

        print(
            "Attempting trusted checkpoint loading..."
        )

        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        except Exception as second_error:

            raise RuntimeError(
                "Could not load MammoSense checkpoint.\n\n"
                f"weights_only=True error:\n"
                f"{first_error}\n\n"
                f"weights_only=False error:\n"
                f"{second_error}"
            )


checkpoint = load_checkpoint(model_path)


# ============================================================
# EXTRACT STATE DICTIONARY
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        # Sometimes the checkpoint itself is a state dict
        state_dict = checkpoint

else:

    raise RuntimeError(
        "Unexpected checkpoint format. "
        "Expected a dictionary/state_dict."
    )


# ============================================================
# CLEAN COMMON PREFIXES
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    new_key = key

    prefixes = [
        "module.",
        "model.",
        "backbone.",
    ]

    changed = True

    while changed:

        changed = False

        for prefix in prefixes:

            if new_key.startswith(prefix):

                new_key = new_key[len(prefix):]

                changed = True

    clean_state_dict[new_key] = value


# ============================================================
# LOAD WEIGHTS
# ============================================================

missing_keys, unexpected_keys = model.load_state_dict(
    clean_state_dict,
    strict=False,
)


print(
    f"Missing keys: {len(missing_keys)}"
)

print(
    f"Unexpected keys: {len(unexpected_keys)}"
)


if len(missing_keys) > 0:

    print("First missing keys:")

    for key in missing_keys[:10]:
        print("  ", key)


if len(unexpected_keys) > 0:

    print("First unexpected keys:")

    for key in unexpected_keys[:10]:
        print("  ", key)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(missing_keys) > 0:

    raise RuntimeError(
        "Model weights were not loaded completely. "
        "The checkpoint architecture may not match "
        "vit_small_patch16_224."
    )


# ============================================================
# DEVICE
# ============================================================

model = model.to(DEVICE)

model.eval()


# ============================================================
# INFERENCE TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
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
# PREDICTION FUNCTION
# ============================================================

@torch.inference_mode()
def predict_image(image: Image.Image):

    # Ensure RGB
    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    logits = model(tensor)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    confidence = float(
        probabilities[predicted_index].item()
    )

    probability_dict = {}

    for i, class_name in enumerate(CLASS_NAMES):

        probability_dict[class_name] = float(
            probabilities[i].item()
        )

    return {
        "prediction": CLASS_NAMES[predicted_index],

        "confidence": confidence,

        "probabilities": probability_dict,
    }


print("=" * 60)
print("MammoSense loaded successfully.")
print("=" * 60)
