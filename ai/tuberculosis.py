# ============================================================
# MEDUSA AI
# TUBERCULOSIS DETECTION
#
# MammoSense TB V12
#
# Architecture:
#   3D ResNet-18
#
# Input:
#   1 x 16 x 224 x 224
#
# Pseudo-3D:
#   2D grayscale X-ray -> replicated 16 times in depth
#
# Classes:
#   0 = NON_TB
#   1 = TB
#
# Hugging Face:
#   Makky07/Tuberculosis
# ============================================================

import io

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

THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL CACHE
# ============================================================

_model = None


# ============================================================
# PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(
            num_output_channels=1
        ),

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
# CREATE MODEL
# ============================================================

def create_model():

    model = r3d_18(
        weights=None,
        progress=False,
    )

    # --------------------------------------------------------
    # Original R3D-18:
    #
    # Conv3d(3, ...)
    #
    # TB V12:
    #
    # Conv3d(1, ...)
    # --------------------------------------------------------

    original_conv = model.stem[0]

    model.stem[0] = nn.Conv3d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )

    # --------------------------------------------------------
    # Binary classifier
    # --------------------------------------------------------

    model.fc = nn.Linear(
        model.fc.in_features,
        1,
    )

    return model


# ============================================================
# CLEAN CHECKPOINT KEYS
# ============================================================

def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global _model

    if _model is not None:

        return _model

    # --------------------------------------------------------
    # Download from Hugging Face
    # --------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
    )

    # --------------------------------------------------------
    # Build architecture
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
    # Metadata confirms:
    #
    # "weights_key": "model_state_dict"
    # --------------------------------------------------------

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Invalid MammoSense TB V12 checkpoint."
        )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # Allow a raw state dictionary as a
        # fallback.

        state_dict = checkpoint

    state_dict = clean_state_dict(
        state_dict
    )

    # --------------------------------------------------------
    # STRICT LOAD
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "MammoSense TB V12 checkpoint could not "
            "be loaded into the expected 3D ResNet-18 "
            "architecture.\n\n"
            f"Checkpoint error:\n{error}"
        ) from error

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    model.to(DEVICE)

    _model = model

    return _model


# ============================================================
# PREPARE X-RAY
# ============================================================

def prepare_image(image):

    # --------------------------------------------------------
    # Accept PIL image
    # --------------------------------------------------------

    if not isinstance(
        image,
        Image.Image,
    ):

        try:

            if isinstance(
                image,
                bytes,
            ):

                image = Image.open(
                    io.BytesIO(image)
                )

            else:

                image = Image.open(
                    image
                )

        except Exception as error:

            raise ValueError(
                "Unable to open the supplied "
                "chest X-ray image."
            ) from error

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    # --------------------------------------------------------
    # Resize + normalize
    # --------------------------------------------------------

    tensor = TRANSFORM(image)

    # Current:
    #
    # [1, 224, 224]
    #
    # channel, height, width

    # --------------------------------------------------------
    # Pseudo-3D replication
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(1)

    # [1, 1, 224, 224]
    #
    # channel, depth, height, width

    tensor = tensor.repeat(
        1,
        DEPTH,
        1,
        1,
    )

    # [1, 16, 224, 224]

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(0)

    # Final:
    #
    # [1, 1, 16, 224, 224]

    return tensor


# ============================================================
# PREDICT
# ============================================================

def predict(image):

    model = load_model()

    tensor = prepare_image(
        image
    ).to(DEVICE)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():

        logits = model(
            tensor
        )

        # Binary classifier:
        #
        # [1, 1]

        logits = logits.reshape(-1)

        tb_probability = torch.sigmoid(
            logits[0]
        ).item()

    # --------------------------------------------------------
    # Numerical safety
    # --------------------------------------------------------

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
    # Standard Medusa output
    # --------------------------------------------------------

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


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info():

    return {

        "model_name":
            "MammoSense TB V12",

        "version":
            "12.0",

        "architecture":
            "3D ResNet-18",

        "task":
            "Tuberculosis detection",

        "domain":
            "Chest X-ray",

        "input_channels":
            1,

        "input_shape":
            [
                1,
                16,
                224,
                224,
            ],

        "classes":
            {
                "0": "NON_TB",
                "1": "TB",
            },

        "positive_class":
            "TB",

        "negative_class":
            "NON_TB",

        "threshold":
            THRESHOLD,

        "device":
            str(DEVICE),

        "repository":
            REPO_ID,

        "checkpoint":
            MODEL_FILENAME,
    }
