# ================================================================
# MEDUSA AI
# TUBERCULOSIS DETECTION MODEL
#
# Hugging Face:
# Makky07/Tuberculosis
#
# Model:
# MammoSense TB V12
#
# Architecture:
# 3D ResNet-18
#
# Input:
# 1 x 16 x 224 x 224
#
# Classes:
# NON_TB
# TB
#
# The original 2D chest X-ray is converted to grayscale,
# resized to 224x224, normalized, then replicated 16 times
# along the depth dimension.
# ================================================================

import io

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from huggingface_hub import hf_hub_download


# ================================================================
# CONFIGURATION
# ================================================================

REPO_ID = "Makky07/Tuberculosis"

MODEL_FILENAME = "mammosense_tb_v12.pt"

IMAGE_SIZE = 224
DEPTH = 16

THRESHOLD = 0.50

CLASS_NAMES = [
    "NON_TB",
    "TB",
]


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# GLOBAL MODEL
# ================================================================

_MODEL = None


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

TRANSFORM = Compose(
    [
        Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        ToTensor(),

        Normalize(
            mean=[0.485],
            std=[0.229],
        ),
    ]
)


# ================================================================
# MODEL ARCHITECTURE
# ================================================================

def create_model():

    model = r3d_18(
        weights=None,
    )

    # ------------------------------------------------------------
    # Original R3D-18 expects RGB input.
    #
    # The TB model expects one grayscale channel.
    # ------------------------------------------------------------

    original_conv = model.stem[0]

    model.stem[0] = nn.Conv3d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )

    # ------------------------------------------------------------
    # Binary classification
    # ------------------------------------------------------------

    model.fc = nn.Linear(
        model.fc.in_features,
        1,
    )

    return model


# ================================================================
# LOAD CHECKPOINT
# ================================================================

def _extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        # Expected format from metadata.json
        if "model_state_dict" in checkpoint:

            return checkpoint["model_state_dict"]

        # Common alternatives
        if "state_dict" in checkpoint:

            return checkpoint["state_dict"]

        if "model" in checkpoint:

            model_value = checkpoint["model"]

            if isinstance(model_value, dict):

                return model_value

    # Raw state_dict
    if isinstance(checkpoint, dict):

        tensor_values = [
            value
            for value in checkpoint.values()
            if torch.is_tensor(value)
        ]

        if tensor_values:

            return checkpoint

    raise RuntimeError(
        "Could not find a valid model_state_dict "
        "inside the tuberculosis checkpoint."
    )


# ================================================================
# CLEAN CHECKPOINT KEYS
# ================================================================

def _clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        # Remove common wrappers
        for prefix in (
            "module.",
            "model.",
            "net.",
        ):

            if new_key.startswith(prefix):

                new_key = new_key[
                    len(prefix):
                ]

        cleaned[new_key] = value

    return cleaned


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    global _MODEL

    if _MODEL is not None:

        return _MODEL

    try:

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
        )

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = _extract_state_dict(
            checkpoint
        )

        state_dict = _clean_state_dict(
            state_dict
        )

        model = create_model()

        # --------------------------------------------------------
        # Strict loading is intentional.
        #
        # If the architecture/checkpoint does not match,
        # fail loudly instead of producing unreliable predictions.
        # --------------------------------------------------------

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model.to(DEVICE)

        model.eval()

        _MODEL = model

        return _MODEL

    except Exception as error:

        raise RuntimeError(
            "Failed to load the Medusa AI tuberculosis model "
            f"from Hugging Face ({REPO_ID}). "
            f"Original error: {error}"
        ) from error


# ================================================================
# PREPARE IMAGE
# ================================================================

def _prepare_image(image):

    if image is None:

        raise ValueError(
            "No medical image was provided."
        )

    # ------------------------------------------------------------
    # Accept PIL Image
    # ------------------------------------------------------------

    if isinstance(image, Image.Image):

        pil_image = image

    # ------------------------------------------------------------
    # Accept raw bytes
    # ------------------------------------------------------------

    elif isinstance(
        image,
        (bytes, bytearray),
    ):

        pil_image = Image.open(
            io.BytesIO(image)
        )

    else:

        raise TypeError(
            "TB prediction expects a PIL Image "
            "or image bytes."
        )

    # ------------------------------------------------------------
    # Convert to grayscale
    # ------------------------------------------------------------

    pil_image = pil_image.convert("L")

    # ------------------------------------------------------------
    # Apply model preprocessing
    # ------------------------------------------------------------

    tensor = TRANSFORM(
        pil_image
    )

    # Current shape:
    #
    # [1, 224, 224]
    #
    # Add depth dimension and replicate 16 times:
    #
    # [1, 16, 224, 224]
    # ------------------------------------------------------------

    tensor = tensor.unsqueeze(1)

    tensor = tensor.repeat(
        1,
        DEPTH,
        1,
        1,
    )

    # Add batch dimension:
    #
    # [1, 1, 16, 224, 224]
    # ------------------------------------------------------------

    tensor = tensor.unsqueeze(0)

    return tensor


# ================================================================
# PREDICTION
# ================================================================

@torch.inference_mode()
def predict(image):

    model = load_model()

    tensor = _prepare_image(
        image
    )

    tensor = tensor.to(
        DEVICE,
        non_blocking=True,
    )

    logits = model(
        tensor
    )

    # ------------------------------------------------------------
    # The model is binary.
    #
    # Output:
    # [batch, 1]
    #
    # Convert logit to TB probability.
    # ------------------------------------------------------------

    if isinstance(
        logits,
        tuple,
    ):

        logits = logits[0]

    if isinstance(
        logits,
        dict,
    ):

        if "logits" in logits:

            logits = logits["logits"]

        else:

            raise RuntimeError(
                "Unexpected dictionary output "
                "from tuberculosis model."
            )

    logits = logits.reshape(-1)

    tb_probability = torch.sigmoid(
        logits[0]
    ).item()

    tb_probability = float(
        max(
            0.0,
            min(
                1.0,
                tb_probability,
            ),
        )
    )

    non_tb_probability = (
        1.0 - tb_probability
    )

    # ------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------

    if tb_probability >= THRESHOLD:

        prediction = "TB"

        confidence = tb_probability

    else:

        prediction = "NON_TB"

        confidence = non_tb_probability

    # ------------------------------------------------------------
    # Return format matches your detection.py
    # ------------------------------------------------------------

    return {
        "prediction": prediction,

        "confidence": float(
            confidence
        ),

        "probabilities": {
            "NON_TB": float(
                non_tb_probability
            ),

            "TB": float(
                tb_probability
            ),
        },
    }


# ================================================================
# OPTIONAL MODEL INFORMATION
# ================================================================

def model_info():

    return {
        "model_name": "MammoSense TB V12",

        "repository": REPO_ID,

        "checkpoint": MODEL_FILENAME,

        "architecture": "3D ResNet-18",

        "input_shape": (
            1,
            DEPTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),

        "classes": CLASS_NAMES,

        "threshold": THRESHOLD,

        "device": str(DEVICE),
    }
