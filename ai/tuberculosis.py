# ================================================================
# MAMMOSENSE TB V12
# TBX11K | PSEUDO-3D RESNET-18
#
# DEPLOYMENT VERSION
#
# Checkpoint source:
#   Hugging Face: Makky07/Tuberculosis
#
# Architecture MUST exactly match training:
#
#   Input X-ray
#       ↓
#   Grayscale
#       ↓
#   224 x 224
#       ↓
#   Normalize: (x - 0.485) / 0.229
#       ↓
#   Replicate 16 times
#       ↓
#   [B, 1, 16, 224, 224]
#       ↓
#   Custom 3D ResNet-18
#       ↓
#   Single logit
#       ↓
#   Sigmoid
#
# Classes:
#   0 = NON_TB
#   1 = TB
#
# ================================================================

from pathlib import Path
from typing import Optional, Dict, Any
import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_NAME = "MammoSense TB V12"
MODEL_VERSION = "12.0"

HF_REPO_ID = "Makky07/Tuberculosis"

IMAGE_SIZE = 224
DEPTH = 16
INPUT_CHANNELS = 1

CLASS_NAMES = [
    "NON_TB",
    "TB",
]

CLASS_TO_IDX = {
    "NON_TB": 0,
    "TB": 1,
}

DEFAULT_THRESHOLD = 0.5


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# MODEL ARCHITECTURE
# ================================================================
#
# THIS MUST MATCH THE TRAINING CODE EXACTLY.
#
# Training architecture:
#
# BasicBlock3D
# ResNet3D18
#
# IMPORTANT:
# We do NOT use torchvision.models.video.r3d_18.
#
# The previous deployment error mentioning:
#
#   VideoResNet
#   layer1.0.conv1.0.weight
#
# came from loading the checkpoint into the WRONG architecture.
#
# This implementation uses:
#
#   layer1.0.conv1.weight
#   layer1.0.bn1.weight
#   layer1.0.conv2.weight
#   layer1.0.bn2.weight
#
# which matches the training script supplied for V12.
#
# ================================================================


class BasicBlock3D(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(
            out_channels
        )

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm3d(
            out_channels
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(

                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),

                nn.BatchNorm3d(
                    out_channels
                ),
            )

        else:

            self.shortcut = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
    ):

        identity = self.shortcut(x)

        out = self.conv1(x)

        out = self.bn1(out)

        out = F.relu(
            out,
            inplace=True,
        )

        out = self.conv2(out)

        out = self.bn2(out)

        out = out + identity

        out = F.relu(
            out,
            inplace=True,
        )

        return out


# ================================================================
# RESNET 3D-18
# ================================================================


class ResNet3D18(nn.Module):

    def __init__(
        self,
        num_classes: int = 1,
    ):
        super().__init__()

        self.in_channels = 64

        # --------------------------------------------------------
        # STEM
        # --------------------------------------------------------

        self.stem = nn.Sequential(

            nn.Conv3d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=(1, 2, 2),
                padding=3,
                bias=False,
            ),

            nn.BatchNorm3d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool3d(
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
            ),
        )

        # --------------------------------------------------------
        # LAYER 1
        # --------------------------------------------------------

        self.layer1 = self._make_layer(
            channels=64,
            blocks=2,
            stride=1,
        )

        # --------------------------------------------------------
        # LAYER 2
        # --------------------------------------------------------

        self.layer2 = self._make_layer(
            channels=128,
            blocks=2,
            stride=2,
        )

        # --------------------------------------------------------
        # LAYER 3
        # --------------------------------------------------------

        self.layer3 = self._make_layer(
            channels=256,
            blocks=2,
            stride=2,
        )

        # --------------------------------------------------------
        # LAYER 4
        # --------------------------------------------------------

        self.layer4 = self._make_layer(
            channels=512,
            blocks=2,
            stride=2,
        )

        # --------------------------------------------------------
        # GLOBAL AVERAGE POOL
        # --------------------------------------------------------

        self.pool = nn.AdaptiveAvgPool3d(
            (1, 1, 1)
        )

        # --------------------------------------------------------
        # CLASSIFIER
        # --------------------------------------------------------

        self.fc = nn.Linear(
            512,
            num_classes,
        )

        self._initialize()

    # ============================================================
    # MAKE LAYER
    # ============================================================

    def _make_layer(
        self,
        channels: int,
        blocks: int,
        stride: int,
    ):

        layers = [
            BasicBlock3D(
                self.in_channels,
                channels,
                stride,
            )
        ]

        self.in_channels = channels

        for _ in range(
            1,
            blocks,
        ):

            layers.append(
                BasicBlock3D(
                    channels,
                    channels,
                )
            )

        return nn.Sequential(
            *layers
        )

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _initialize(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv3d,
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                module,
                nn.BatchNorm3d,
            ):

                nn.init.constant_(
                    module.weight,
                    1,
                )

                nn.init.constant_(
                    module.bias,
                    0,
                )

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        x: torch.Tensor,
    ):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        return self.fc(x)


# ================================================================
# GLOBAL MODEL CACHE
# ================================================================

_MODEL = None
_MODEL_PATH = None


# ================================================================
# LOCAL MODEL PATHS
# ================================================================


def _possible_model_paths():

    current_file = Path(
        __file__
    ).resolve()

    project_root = (
        current_file.parent.parent
    )

    paths = [

        project_root
        / "models"
        / "mammosense_tb_v12.pt",

        project_root
        / "model"
        / "mammosense_tb_v12.pt",

        current_file.parent
        / "mammosense_tb_v12.pt",

        current_file.parent
        / "mammosense_tb_v12"
        / "mammosense_tb_v12.pt",

        project_root
        / "mammosense_tb_v12.pt",

        project_root
        / "models"
        / "mammosense_tb_v12 (1).pt",

        project_root
        / "models"
        / "mammosense_tb_v12 (2).pt",

        Path(
            "/tmp/mammosense_tb_v12.pt"
        ),

        Path(
            "/app/mammosense_tb_v12.pt"
        ),

        Path(
            "/mount/src/medusa/mammosense_tb_v12.pt"
        ),

        Path(
            "/mount/src/medusa/models/mammosense_tb_v12.pt"
        ),
    ]

    unique = []

    seen = set()

    for path in paths:

        path = Path(path)

        key = str(path)

        if key not in seen:

            seen.add(key)

            unique.append(path)

    return unique


# ================================================================
# HUGGING FACE CHECKPOINT DOWNLOAD
# ================================================================
#
# We dynamically inspect the repository instead of assuming that
# the filename is exactly "mammosense_tb_v12.pt".
#
# This protects deployment if the uploaded file has a name such as:
#
#   mammosense_tb_v12 (1).pt
#
# ================================================================


def _download_from_huggingface() -> Path:

    cache_dir = Path(
        os.environ.get(
            "MEDUSA_MODEL_CACHE",
            "/tmp/medusa_models",
        )
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        from huggingface_hub import (
            HfApi,
            hf_hub_download,
        )

    except ImportError as error:

        raise RuntimeError(
            "huggingface_hub is required to download "
            "the MammoSense TB V12 model.\n\n"
            "Add this package to requirements.txt:\n"
            "huggingface_hub"
        ) from error

    # ------------------------------------------------------------
    # Get repository file list
    # ------------------------------------------------------------

    try:

        api = HfApi()

        files = api.list_repo_files(
            repo_id=HF_REPO_ID,
            repo_type="model",
        )

    except Exception as error:

        raise RuntimeError(
            "Could not access the MammoSense TB V12 "
            "Hugging Face repository.\n\n"
            f"Repository: {HF_REPO_ID}\n"
            f"Error: {error}"
        ) from error

    # ------------------------------------------------------------
    # Find PyTorch checkpoint files
    # ------------------------------------------------------------

    checkpoint_candidates = [
        filename
        for filename in files
        if filename.lower().endswith(
            (
                ".pt",
                ".pth",
                ".bin",
            )
        )
    ]

    if not checkpoint_candidates:

        raise FileNotFoundError(
            "No PyTorch checkpoint was found in the "
            f"Hugging Face repository: {HF_REPO_ID}\n\n"
            "Repository files:\n"
            + "\n".join(
                f"  - {name}"
                for name in files
            )
        )

    # ------------------------------------------------------------
    # Prefer V12 filename
    # ------------------------------------------------------------

    preferred = []

    for filename in checkpoint_candidates:

        lower = filename.lower()

        if (
            "v12" in lower
            and (
                lower.endswith(".pt")
                or lower.endswith(".pth")
            )
        ):

            preferred.append(
                filename
            )

    if preferred:

        # Prefer the shortest matching V12 filename.

        filename = sorted(
            preferred,
            key=len,
        )[0]

    else:

        # If no V12 filename exists, use the first
        # PyTorch checkpoint.

        filename = sorted(
            checkpoint_candidates,
            key=len,
        )[0]

    # ------------------------------------------------------------
    # Download
    # ------------------------------------------------------------

    try:

        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="model",
            cache_dir=str(cache_dir),
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to download the MammoSense TB V12 "
            "checkpoint from Hugging Face.\n\n"
            f"Repository: {HF_REPO_ID}\n"
            f"Selected file: {filename}\n"
            f"Error: {error}"
        ) from error

    downloaded_path = Path(
        downloaded
    )

    if not downloaded_path.exists():

        raise FileNotFoundError(
            "Hugging Face reported a successful download, "
            "but the checkpoint file does not exist locally:\n"
            f"{downloaded_path}"
        )

    if downloaded_path.stat().st_size < 1_000_000:

        raise RuntimeError(
            "Downloaded MammoSense TB V12 checkpoint "
            "is suspiciously small.\n\n"
            f"File: {downloaded_path}\n"
            f"Size: {downloaded_path.stat().st_size} bytes"
        )

    return downloaded_path


# ================================================================
# FIND MODEL
# ================================================================


def find_model_path(
    explicit_path: Optional[str] = None,
) -> Path:

    # ------------------------------------------------------------
    # 1. Explicit path
    # ------------------------------------------------------------

    if explicit_path:

        path = Path(
            explicit_path
        )

        if path.exists() and path.is_file():

            return path

        raise FileNotFoundError(
            "MammoSense TB model was not found at:\n"
            f"{path}"
        )

    # ------------------------------------------------------------
    # 2. Local model
    # ------------------------------------------------------------

    for path in _possible_model_paths():

        if (
            path.exists()
            and path.is_file()
        ):

            return path

    # ------------------------------------------------------------
    # 3. Hugging Face
    # ------------------------------------------------------------

    return _download_from_huggingface()


# ================================================================
# EXTRACT STATE DICT
# ================================================================


def _extract_state_dict(
    checkpoint: Any,
) -> Dict[str, torch.Tensor]:

    if isinstance(
        checkpoint,
        dict,
    ):

        # --------------------------------------------------------
        # Exact V12 format
        # --------------------------------------------------------

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = checkpoint[
                "model_state_dict"
            ]

            if isinstance(
                state_dict,
                dict,
            ):

                return state_dict

        # --------------------------------------------------------
        # Common alternatives
        # --------------------------------------------------------

        for key in (
            "state_dict",
            "model",
            "weights",
        ):

            if key in checkpoint:

                candidate = checkpoint[key]

                if isinstance(
                    candidate,
                    dict,
                ):

                    if any(
                        isinstance(
                            value,
                            torch.Tensor,
                        )
                        for value in candidate.values()
                    ):

                        return candidate

        # --------------------------------------------------------
        # Raw state dictionary
        # --------------------------------------------------------

        if checkpoint:

            if all(
                isinstance(
                    key,
                    str,
                )
                for key in checkpoint.keys()
            ):

                if all(
                    isinstance(
                        value,
                        torch.Tensor,
                    )
                    for value in checkpoint.values()
                ):

                    return checkpoint

    raise RuntimeError(
        "The MammoSense TB V12 checkpoint does not "
        "contain a recognizable PyTorch state_dict."
    )


# ================================================================
# CLEAN STATE DICT PREFIXES
# ================================================================


def _clean_state_dict(
    state_dict: Dict[str, torch.Tensor],
):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        # DataParallel

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                7:
            ]

        # Some training wrappers

        if new_key.startswith(
            "model."
        ):

            new_key = new_key[
                6:
            ]

        cleaned[new_key] = value

    return cleaned


# ================================================================
# LOAD MODEL
# ================================================================


def load_model(
    model_path: Optional[str] = None,
):

    global _MODEL
    global _MODEL_PATH

    # ------------------------------------------------------------
    # Cached model
    # ------------------------------------------------------------

    if (
        _MODEL is not None
        and model_path is None
    ):

        return _MODEL

    # ------------------------------------------------------------
    # Locate checkpoint
    # ------------------------------------------------------------

    try:

        checkpoint_path = find_model_path(
            explicit_path=model_path
        )

    except Exception as error:

        raise RuntimeError(
            "MammoSense TB V12 checkpoint could not "
            "be located.\n\n"
            f"Hugging Face repository: {HF_REPO_ID}\n\n"
            f"Error: {error}"
        ) from error

    # ------------------------------------------------------------
    # Build EXACT architecture
    # ------------------------------------------------------------

    model = ResNet3D18(
        num_classes=1
    )

    # ------------------------------------------------------------
    # Read checkpoint
    # ------------------------------------------------------------

    try:

        try:

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        except TypeError:

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

    except Exception as error:

        raise RuntimeError(
            "MammoSense TB V12 checkpoint could not "
            "be read.\n\n"
            f"Checkpoint: {checkpoint_path}\n\n"
            f"Error: {error}"
        ) from error

    # ------------------------------------------------------------
    # Extract state dictionary
    # ------------------------------------------------------------

    try:

        state_dict = _extract_state_dict(
            checkpoint
        )

        state_dict = _clean_state_dict(
            state_dict
        )

    except Exception as error:

        raise RuntimeError(
            "MammoSense TB V12 checkpoint does not "
            "contain a valid model state dictionary.\n\n"
            f"Checkpoint: {checkpoint_path}\n\n"
            f"Error: {error}"
        ) from error

    # ------------------------------------------------------------
    # Compare architecture BEFORE loading
    # ------------------------------------------------------------

    expected_state = model.state_dict()

    expected_keys = set(
        expected_state.keys()
    )

    received_keys = set(
        state_dict.keys()
    )

    missing = sorted(
        expected_keys - received_keys
    )

    unexpected = sorted(
        received_keys - expected_keys
    )

    # ------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------

    shape_errors = []

    for key in sorted(
        expected_keys.intersection(
            received_keys
        )
    ):

        expected_shape = tuple(
            expected_state[key].shape
        )

        received_shape = tuple(
            state_dict[key].shape
        )

        if expected_shape != received_shape:

            shape_errors.append(
                (
                    key,
                    expected_shape,
                    received_shape,
                )
            )

    # ------------------------------------------------------------
    # Fail clearly if checkpoint is incompatible
    # ------------------------------------------------------------

    if (
        missing
        or unexpected
        or shape_errors
    ):

        missing_text = "\n".join(
            f"  - {key}"
            for key in missing[:25]
        )

        unexpected_text = "\n".join(
            f"  - {key}"
            for key in unexpected[:25]
        )

        shape_text = "\n".join(
            (
                f"  - {key}: "
                f"expected {expected_shape}, "
                f"received {received_shape}"
            )
            for key, expected_shape, received_shape
            in shape_errors[:15]
        )

        raise RuntimeError(
            "MammoSense TB V12 checkpoint is "
            "INCOMPATIBLE with the V12 3D ResNet-18 "
            "architecture.\n\n"

            f"Checkpoint: {checkpoint_path}\n\n"

            f"Missing keys: {len(missing)}\n"
            f"{missing_text if missing_text else '  None'}\n\n"

            f"Unexpected keys: {len(unexpected)}\n"
            f"{unexpected_text if unexpected_text else '  None'}\n\n"

            f"Shape mismatches: {len(shape_errors)}\n"
            f"{shape_text if shape_text else '  None'}\n\n"

            "This means the uploaded checkpoint was "
            "not trained with the exact ResNet3D18 "
            "architecture in this file."
        )

    # ------------------------------------------------------------
    # Strict loading
    # ------------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "MammoSense TB V12 checkpoint failed "
            "strict state_dict loading.\n\n"
            f"Checkpoint: {checkpoint_path}\n\n"
            f"Error:\n{error}"
        ) from error

    # ------------------------------------------------------------
    # Move model
    # ------------------------------------------------------------

    model = model.to(
        DEVICE
    )

    model.eval()

    # ------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------

    _MODEL = model
    _MODEL_PATH = checkpoint_path

    return model


# ================================================================
# PREPROCESS IMAGE
# ================================================================


def preprocess_image(
    image,
) -> torch.Tensor:

    # ------------------------------------------------------------
    # Accept PIL image or file-like/path
    # ------------------------------------------------------------

    if not isinstance(
        image,
        Image.Image,
    ):

        try:

            image = Image.open(
                image
            )

        except Exception as error:

            raise ValueError(
                "Could not open the supplied "
                "chest X-ray image."
            ) from error

    # ------------------------------------------------------------
    # Grayscale
    # ------------------------------------------------------------

    image = image.convert(
        "L"
    )

    # ------------------------------------------------------------
    # Exact training resize
    # ------------------------------------------------------------

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        Image.Resampling.BILINEAR,
    )

    # ------------------------------------------------------------
    # Convert to NumPy
    # ------------------------------------------------------------

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # 0-255 -> 0-1
    # ------------------------------------------------------------

    array = array / 255.0

    # ------------------------------------------------------------
    # EXACT training normalization
    # ------------------------------------------------------------

    array = (
        array - 0.485
    ) / 0.229

    # ------------------------------------------------------------
    # [H,W]
    # ->
    # [1,H,W]
    # ------------------------------------------------------------

    tensor = torch.from_numpy(
        array
    )

    tensor = tensor.unsqueeze(
        0
    )

    # ------------------------------------------------------------
    # [1,H,W]
    # ->
    # [1,1,H,W]
    # ------------------------------------------------------------

    tensor = tensor.unsqueeze(
        1
    )

    # ------------------------------------------------------------
    # Replicate along depth
    #
    # [1,1,H,W]
    # ->
    # [1,16,H,W]
    # ------------------------------------------------------------

    tensor = tensor.repeat(
        1,
        DEPTH,
        1,
        1,
    )

    # ------------------------------------------------------------
    # Add batch dimension
    #
    # [1,16,H,W]
    # ->
    # [1,1,16,H,W]
    # ------------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    return tensor.float()


# ================================================================
# PREDICT
# ================================================================


def predict(
    image,
    threshold: float = DEFAULT_THRESHOLD,
    model=None,
):

    if model is None:

        model = load_model()

    # ------------------------------------------------------------
    # Validate threshold
    # ------------------------------------------------------------

    threshold = float(
        threshold
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    # ------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------

    tensor = preprocess_image(
        image
    )

    tensor = tensor.to(
        DEVICE,
        non_blocking=True,
    )

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------

    model.eval()

    with torch.inference_mode():

        logits = model(
            tensor
        )

        logits = logits.reshape(
            -1
        )

        probability = torch.sigmoid(
            logits[0]
        ).item()

    # ------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------

    predicted_label = int(
        probability >= threshold
    )

    predicted_class = (
        "TB"
        if predicted_label == 1
        else "NON_TB"
    )

    non_tb_probability = (
        1.0 - probability
    )

    confidence = max(
        probability,
        non_tb_probability,
    )

    # ------------------------------------------------------------
    # Return result
    # ------------------------------------------------------------

    return {

        "prediction":
            predicted_class,

        "predicted_class":
            predicted_class,

        "predicted_label":
            predicted_label,

        "tb_probability":
            float(probability),

        "non_tb_probability":
            float(non_tb_probability),

        "confidence":
            float(confidence),

        "threshold":
            threshold,

        "class_names":
            CLASS_NAMES,

        "class_to_idx":
            CLASS_TO_IDX,

        "model_name":
            MODEL_NAME,

        "model_version":
            MODEL_VERSION,

        "architecture":
            "3D ResNet-18",

        "dataset":
            "TBX11K",

        "input_shape":
            [
                1,
                INPUT_CHANNELS,
                DEPTH,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ],

        "device":
            str(DEVICE),

        "checkpoint":
            str(_MODEL_PATH)
            if _MODEL_PATH is not None
            else None,
    }


# ================================================================
# MEDUSA COMPATIBILITY
# ================================================================


def load_tb_model(
    model_path: Optional[str] = None,
):

    return load_model(
        model_path=model_path
    )


def predict_tb(
    image,
    threshold: float = DEFAULT_THRESHOLD,
    model=None,
):

    return predict(
        image=image,
        threshold=threshold,
        model=model,
    )


# ================================================================
# MODEL INFORMATION
# ================================================================


def get_model_info():

    return {

        "model_name":
            MODEL_NAME,

        "version":
            MODEL_VERSION,

        "architecture":
            "3D ResNet-18",

        "dataset":
            "TBX11K",

        "task":
            "Binary tuberculosis classification",

        "classes":
            CLASS_NAMES,

        "class_to_idx":
            CLASS_TO_IDX,

        "input_channels":
            INPUT_CHANNELS,

        "image_size":
            IMAGE_SIZE,

        "depth":
            DEPTH,

        "pseudo_3d":
            True,

        "input_shape":
            [
                INPUT_CHANNELS,
                DEPTH,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ],

        "normalization_mean":
            0.485,

        "normalization_std":
            0.229,

        "threshold":
            DEFAULT_THRESHOLD,

        "huggingface_repo":
            HF_REPO_ID,

        "device":
            str(DEVICE),
    }


# ================================================================
# VERIFY MODEL
# ================================================================


def verify_model(
    model_path: Optional[str] = None,
):

    model = load_model(
        model_path=model_path
    )

    # ------------------------------------------------------------
    # Exact input expected by model
    # ------------------------------------------------------------

    dummy = torch.zeros(
        (
            1,
            INPUT_CHANNELS,
            DEPTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    # ------------------------------------------------------------
    # Forward test
    # ------------------------------------------------------------

    with torch.inference_mode():

        output = model(
            dummy
        )

    # ------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------

    if tuple(
        output.shape
    ) != (1, 1):

        raise RuntimeError(
            "MammoSense TB V12 loaded, but produced "
            "an unexpected output shape:\n"
            f"Expected: (1, 1)\n"
            f"Received: {tuple(output.shape)}"
        )

    probability = torch.sigmoid(
        output[0, 0]
    ).item()

    return {

        "loaded":
            True,

        "checkpoint":
            str(_MODEL_PATH),

        "device":
            str(DEVICE),

        "input_shape":
            [
                1,
                INPUT_CHANNELS,
                DEPTH,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ],

        "output_shape":
            list(output.shape),

        "sample_probability":
            float(probability),
    }


# ================================================================
# OPTIONAL STARTUP TEST
# ================================================================


if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "MAMMOSENSE TB V12"
    )

    print(
        "TBX11K | Pseudo-3D ResNet-18"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Hugging Face: {HF_REPO_ID}"
    )

    print(
        "=" * 72
    )

    result = verify_model()

    print(
        "\nMODEL VERIFIED"
    )

    print(
        f"Checkpoint: {result['checkpoint']}"
    )

    print(
        f"Input: {result['input_shape']}"
    )

    print(
        f"Output: {result['output_shape']}"
    )

    print(
        f"Sample TB probability: "
        f"{result['sample_probability']:.6f}"
    )

    print(
        "\n✓ MammoSense TB V12 is ready."
    )
