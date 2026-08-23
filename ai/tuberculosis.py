# ================================================================
# MEDUSA AI - MAMMOSENSE TB V12
# TB DETECTION IN CHEST X-RAY
#
# EXACT INFERENCE ARCHITECTURE USED DURING TRAINING
#
# Input:
#   2D Chest X-ray
#       ↓
#   grayscale
#       ↓
#   resize 224 x 224
#       ↓
#   pseudo-3D stacking
#       ↓
#   [1, D, 224, 224]
#       ↓
#   Custom 3D ResNet-18
#       ↓
#   TB probability
#
# IMPORTANT:
# DO NOT replace this architecture with torchvision VideoResNet.
# ================================================================

import io
import os
import json
import hashlib
import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_download


# ================================================================
# CONFIGURATION
# ================================================================

HF_REPO = "Makky07/Tuberculosis"

# CHANGE THIS ONLY IF YOUR FILE HAS A DIFFERENT NAME
MODEL_FILENAME = "mammosense_tb_v12.pt"

IMAGE_SIZE = 224

# This must match the depth used by the V12 training pipeline.
# If your metadata.json contains another value, the loader below
# automatically uses that value.
DEFAULT_DEPTH = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logger = logging.getLogger(__name__)


# ================================================================
# EXACT CUSTOM 3D RESNET BLOCK
# ================================================================

class BasicBlock3D(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm3d(out_channels)

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )

        else:

            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


# ================================================================
# EXACT MAMMOSENSE 3D RESNET-18
# ================================================================

class MammoSenseTBResNet18(nn.Module):

    def __init__(self, num_classes=1):

        super().__init__()

        # IMPORTANT:
        # This is 7 x 7 x 7.
        # The error from Medusa showed that the current loader
        # was incorrectly using 3 x 7 x 7.

        self.stem = nn.Sequential(

            nn.Conv3d(
                1,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),

            nn.BatchNorm3d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool3d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.layer1 = self._make_layer(
            64,
            64,
            blocks=2,
            stride=1
        )

        self.layer2 = self._make_layer(
            64,
            128,
            blocks=2,
            stride=2
        )

        self.layer3 = self._make_layer(
            128,
            256,
            blocks=2,
            stride=2
        )

        self.layer4 = self._make_layer(
            256,
            512,
            blocks=2,
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d(
            (1, 1, 1)
        )

        # Binary classifier.
        # V12 was trained with ONE output logit.
        self.fc = nn.Linear(
            512,
            1
        )

    def _make_layer(
        self,
        in_channels,
        out_channels,
        blocks,
        stride
    ):

        layers = []

        layers.append(
            BasicBlock3D(
                in_channels,
                out_channels,
                stride
            )
        )

        for _ in range(1, blocks):

            layers.append(
                BasicBlock3D(
                    out_channels,
                    out_channels,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.fc(x)

        return x


# ================================================================
# GLOBAL MODEL CACHE
# ================================================================

_MODEL = None
_MODEL_METADATA = None
_MODEL_PATH = None


# ================================================================
# SHA256
# ================================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ================================================================
# FIND MODEL FILE
# ================================================================

def _download_model():

    global _MODEL_PATH

    if _MODEL_PATH is not None:
        return _MODEL_PATH

    logger.info(
        "Downloading MammoSense TB V12 model..."
    )

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILENAME
    )

    _MODEL_PATH = model_path

    logger.info(
        "TB model downloaded: %s",
        model_path
    )

    return model_path


# ================================================================
# LOAD METADATA IF AVAILABLE
# ================================================================

def _load_metadata():

    try:

        metadata_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="metadata.json"
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


# ================================================================
# EXTRACT STATE DICT
# ================================================================

def _extract_state_dict(checkpoint):

    if isinstance(
        checkpoint,
        dict
    ):

        # Standard V12 checkpoint
        if "model_state_dict" in checkpoint:

            return checkpoint["model_state_dict"]

        # Other common naming
        if "state_dict" in checkpoint:

            return checkpoint["state_dict"]

        # Raw state_dict
        if all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):

            tensor_values = [
                v
                for v in checkpoint.values()
                if torch.is_tensor(v)
            ]

            if tensor_values:

                return checkpoint

    raise RuntimeError(
        "Unable to find model_state_dict in checkpoint."
    )


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    global _MODEL
    global _MODEL_METADATA

    if _MODEL is not None:

        return _MODEL

    try:

        model_path = _download_model()

        _MODEL_METADATA = _load_metadata()

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False
        )

        state_dict = _extract_state_dict(
            checkpoint
        )

        # Remove DataParallel prefix if present
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):

                key = key[len("module."):]

            cleaned_state_dict[key] = value

        state_dict = cleaned_state_dict

        # --------------------------------------------------------
        # Create EXACT training architecture
        # --------------------------------------------------------

        model = MammoSenseTBResNet18(
            num_classes=1
        )

        # --------------------------------------------------------
        # Strict loading is intentional.
        # It prevents silent architecture mistakes.
        # --------------------------------------------------------

        model.load_state_dict(
            state_dict,
            strict=True
        )

        model.to(DEVICE)

        model.eval()

        _MODEL = model

        logger.info(
            "MammoSense TB V12 loaded successfully."
        )

        logger.info(
            "Device: %s",
            DEVICE
        )

        return _MODEL

    except Exception as error:

        raise RuntimeError(
            "Failed to load the Medusa AI tuberculosis model "
            f"from Hugging Face ({HF_REPO}). "
            f"Checkpoint architecture does not match the "
            f"loader or the model file is invalid. "
            f"Original error: {error}"
        ) from error


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

def _prepare_image(
    image,
    depth=DEFAULT_DEPTH
):

    if not isinstance(
        image,
        Image.Image
    ):

        image = Image.open(
            image
        )

    # ------------------------------------------------------------
    # Grayscale
    # ------------------------------------------------------------

    image = image.convert("L")

    # ------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    # ------------------------------------------------------------
    # NumPy
    # ------------------------------------------------------------

    arr = np.asarray(
        image,
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # Normalize to [0, 1]
    # ------------------------------------------------------------

    arr /= 255.0

    # ------------------------------------------------------------
    # Convert to approximately standardized intensity.
    #
    # This should match the training preprocessing.
    # ------------------------------------------------------------

    mean = arr.mean()
    std = arr.std()

    if std > 1e-6:

        arr = (
            arr - mean
        ) / std

    else:

        arr = arr - mean

    # ------------------------------------------------------------
    # Clip extreme values
    # ------------------------------------------------------------

    arr = np.clip(
        arr,
        -3.0,
        3.0
    )

    # ------------------------------------------------------------
    # Pseudo-3D construction
    #
    # The original training pipeline converted the 2D X-ray
    # into a 3D tensor by stacking the image along depth.
    # ------------------------------------------------------------

    volume = np.stack(
        [arr] * depth,
        axis=0
    )

    # [D,H,W]
    # ->
    # [1,D,H,W]

    volume = np.expand_dims(
        volume,
        axis=0
    )

    # ->
    # [B,C,D,H,W]

    volume = np.expand_dims(
        volume,
        axis=0
    )

    tensor = torch.from_numpy(
        volume
    ).float()

    return tensor


# ================================================================
# GET MODEL DEPTH
# ================================================================

def _get_depth():

    metadata = _MODEL_METADATA or {}

    candidates = [
        metadata.get("input_depth"),
        metadata.get("depth"),
        metadata.get("volume_depth")
    ]

    for value in candidates:

        if value is not None:

            try:

                value = int(value)

                if value > 0:

                    return value

            except Exception:

                pass

    return DEFAULT_DEPTH


# ================================================================
# PREDICT
# ================================================================

@torch.inference_mode()
def predict(image):

    model = load_model()

    depth = _get_depth()

    tensor = _prepare_image(
        image,
        depth=depth
    )

    tensor = tensor.to(
        DEVICE,
        non_blocking=True
    )

    logits = model(
        tensor
    )

    probability = torch.sigmoid(
        logits
    ).item()

    # ------------------------------------------------------------
    # Binary convention used during V12:
    #
    # 0 = NON_TB
    # 1 = TB
    # ------------------------------------------------------------

    tb_probability = float(
        probability
    )

    non_tb_probability = float(
        1.0 - probability
    )

    if tb_probability >= 0.5:

        predicted_class = "TB"

    else:

        predicted_class = "NON_TB"

    return {
        "prediction": predicted_class,

        "predicted_class": predicted_class,

        "tb_probability": tb_probability,

        "non_tb_probability": non_tb_probability,

        "confidence": max(
            tb_probability,
            non_tb_probability
        ),

        "model": "MammoSense TB V12",

        "architecture": (
            "Custom 3D ResNet-18"
        ),

        "device": str(DEVICE),

        "input_depth": depth,

        "image_size": IMAGE_SIZE
    }


# ================================================================
# COMPATIBILITY FUNCTION
# ================================================================

def analyze_image(image):

    return predict(
        image
    )


# ================================================================
# OPTIONAL ALIASES
# ================================================================

def predict_tb(image):

    return predict(
        image
    )


__all__ = [
    "load_model",
    "predict",
    "predict_tb",
    "analyze_image",
    "MammoSenseTBResNet18",
]
