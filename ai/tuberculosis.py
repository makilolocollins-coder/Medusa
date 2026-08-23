# ============================================================
# MEDUSA AI
# TUBERCULOSIS MODEL
#
# MammoSense TB V12
#
# Architecture:
#   3D ResNet-18
#   Input: 1 x 16 x 224 x 224
#
# Preprocessing:
#   Grayscale
#   Resize: 224 x 224
#   Replicate image 16 times along depth
#   Normalize: mean=0.485, std=0.229
#
# Classes:
#   0 = NON_TB
#   1 = TB
#
# Hugging Face:
#   Makky07/Tuberculosis
# ============================================================

import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models.video import r3d_18
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = "Makky07/Tuberculosis"

MODEL_FILENAME = "mammosense_tb_v12.pt"

IMAGE_SIZE = 224
DEPTH = 16

CLASS_NAMES = {
    0: "NON_TB",
    1: "TB",
}

THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# GLOBAL MODEL
# ============================================================

_model = None


# ============================================================
# PREPROCESSING
# ============================================================

_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485],
            std=[0.229],
        ),
    ]
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def create_model():
    """
    Create the 3D ResNet-18 architecture used by
    MammoSense TB V12.

    The original r3d_18 expects 3 channels.
    TB V12 uses a single grayscale channel.
    """

    model = r3d_18(
        weights=None,
        progress=False,
    )

    # --------------------------------------------------------
    # Change first convolution from 3 channels -> 1 channel
    # --------------------------------------------------------

    old_conv = model.stem[0]

    new_conv = nn.Conv3d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    model.stem[0] = new_conv

    # --------------------------------------------------------
    # Binary classification
    # --------------------------------------------------------

    model.fc = nn.Linear(
        model.fc.in_features,
        1,
    )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def _clean_state_dict(state_dict):
    """
    Remove common prefixes such as 'module.'
    from DataParallel checkpoints.
    """

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    return cleaned


def load_model():
    """
    Download and load MammoSense TB V12 from Hugging Face.

    The model is loaded only once and cached in memory.
    """

    global _model

    if _model is not None:
        return _model

    # --------------------------------------------------------
    # Download checkpoint
    # --------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
    )

    # --------------------------------------------------------
    # Create architecture
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            # Some checkpoints are themselves
            # state dictionaries.

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    state_dict = _clean_state_dict(
        state_dict
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "MammoSense TB V12 checkpoint does not "
            "match the expected 3D ResNet-18 architecture. "
            f"Original error: {error}"
        ) from error

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model = model.to(DEVICE)
    model.eval()

    _model = model

    return _model


# ============================================================
# IMAGE PREPARATION
# ============================================================

def _prepare_image(image):
    """
    Convert a PIL image into the pseudo-3D tensor expected
    by MammoSense TB V12.

    Final tensor:

        [1, 1, 16, 224, 224]

    Meaning:

        batch
        channel
        depth
        height
        width
    """

    if not isinstance(image, Image.Image):

        try:
            image = Image.open(image)

        except Exception as error:

            raise ValueError(
                "The supplied image could not be opened."
            ) from error

    # --------------------------------------------------------
    # Ensure RGB/PIL compatibility
    # --------------------------------------------------------

    image = image.convert("RGB")

    # --------------------------------------------------------
    # Standard preprocessing
    # --------------------------------------------------------

    tensor = _transform(image)

    # Current shape:
    #
    # [1, 224, 224]

    # --------------------------------------------------------
    # Replicate 2D X-ray 16 times along depth
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(1)

    # [1, 1, 224, 224]
    #
    # -> [1, 16, 224, 224]

    tensor = tensor.repeat(
        1,
        DEPTH,
        1,
        1,
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(0)

    # Final:
    #
    # [1, 1, 16, 224, 224]

    return tensor


# ============================================================
# PREDICTION
# ============================================================

def predict(image):
    """
    Run tuberculosis prediction.

    Returns:

    {
        "prediction": "TB" or "NON_TB",
        "confidence": float,
        "probabilities": {
            "NON_TB": float,
            "TB": float
        }
    }
    """

    model = load_model()

    tensor = _prepare_image(image)

    tensor = tensor.to(DEVICE)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(tensor)

        # Expected shape:
        # [1, 1]

        logits = logits.reshape(-1)

        tb_probability = torch.sigmoid(
            logits[0]
        ).item()

    # --------------------------------------------------------
    # Clamp numerical values
    # --------------------------------------------------------

    tb_probability = max(
        0.0,
        min(
            1.0,
            float(tb_probability),
        ),
    )

    non_tb_probability = (
        1.0 - tb_probability
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if tb_probability >= THRESHOLD:

        prediction = "TB"
        confidence = tb_probability

    else:

        prediction = "NON_TB"
        confidence = non_tb_probability

    # --------------------------------------------------------
    # Return standard Medusa result format
    # --------------------------------------------------------

    return {
        "prediction": prediction,

        "confidence": confidence,

        "probabilities": {
            "NON_TB": non_tb_probability,
            "TB": tb_probability,
        },
    }


# ============================================================
# OPTIONAL MODEL INFORMATION
# ============================================================

def get_model_info():

    return {
        "model_name": "MammoSense TB V12",
        "version": "12.0",
        "architecture": "3D ResNet-18",
        "task": "Tuberculosis detection",
        "input_shape": [
            1,
            16,
            224,
            224,
        ],
        "classes": [
            "NON_TB",
            "TB",
        ],
        "threshold": THRESHOLD,
        "device": str(DEVICE),
        "huggingface_repo": REPO_ID,
    }
