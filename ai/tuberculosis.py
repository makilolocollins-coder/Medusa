# ================================================================
# MEDUSA AI
# MAMMOSENSE TB V13
#
# TBX11K | CUSTOM 3D RESNET-18
#
# Hugging Face:
# Makky07/Tuberculosis_
#
# Checkpoint:
# mammosense_tb_v13.pt
#
# INPUT:
# Grayscale chest X-ray
# -> 224 x 224
# -> normalize
# -> replicate 16 times
# -> [B, 1, 16, 224, 224]
#
# OUTPUT:
# NORMAL = 0
# TB     = 1
# ================================================================

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_NAME = "MammoSense TB V13"
MODEL_VERSION = "13.0"

HF_REPO_ID = "Makky07/Tuberculosis_"
HF_FILENAME = "mammosense_tb_v13.pt"

ARCHITECTURE = "Custom 3D ResNet-18"

IMAGE_SIZE = 224
INPUT_DEPTH = 16
INPUT_CHANNELS = 1

CLASS_NAMES = [
    "NORMAL",
    "TB",
]

CLASS_TO_IDX = {
    "NORMAL": 0,
    "TB": 1,
}

DEFAULT_THRESHOLD = 0.5


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# 3D BASIC BLOCK
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
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(
            out_channels
        )

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
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
                    in_channels,
                    out_channels,
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

    def forward(self, x):

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
# EXACT V13 3D RESNET-18
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
                1,
                64,
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
        # LAYERS
        # --------------------------------------------------------

        self.layer1 = self._make_layer(
            64,
            2,
            stride=1,
        )

        self.layer2 = self._make_layer(
            128,
            2,
            stride=2,
        )

        self.layer3 = self._make_layer(
            256,
            2,
            stride=2,
        )

        self.layer4 = self._make_layer(
            512,
            2,
            stride=2,
        )

        # --------------------------------------------------------
        # GLOBAL POOL
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

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1,
        )

        return self.fc(x)


# ================================================================
# CHECKPOINT CACHE
# ================================================================

_MODEL = None
_MODEL_PATH = None
_MODEL_METADATA = None


# ================================================================
# DOWNLOAD V13 FROM HUGGING FACE
# ================================================================

def download_model() -> Path:

    if hf_hub_download is None:

        raise RuntimeError(
            "huggingface_hub is not installed.\n\n"
            "Add this to requirements.txt:\n"
            "huggingface_hub"
        )

    try:

        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
        )

    except Exception as error:

        raise RuntimeError(
            "Could not download the MammoSense TB V13 "
            "checkpoint from Hugging Face.\n\n"
            f"Repository: {HF_REPO_ID}\n"
            f"File: {HF_FILENAME}\n\n"
            f"Error: {error}"
        ) from error

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            "Hugging Face reported a checkpoint path, "
            "but the file does not exist:\n"
            f"{path}"
        )

    return path


# ================================================================
# FIND MODEL
# ================================================================

def find_model_path(
    explicit_path: Optional[str] = None,
) -> Path:

    # ------------------------------------------------------------
    # Explicit local path
    # ------------------------------------------------------------

    if explicit_path:

        path = Path(
            explicit_path
        )

        if path.exists() and path.is_file():

            return path

        raise FileNotFoundError(
            "MammoSense TB V13 model was not found at:\n"
            f"{path}"
        )

    # ------------------------------------------------------------
    # Optional local project locations
    # ------------------------------------------------------------

    current_file = Path(
        __file__
    ).resolve()

    project_root = (
        current_file.parent.parent
    )

    local_paths = [

        project_root
        / "models"
        / HF_FILENAME,

        project_root
        / "model"
        / HF_FILENAME,

        current_file.parent
        / HF_FILENAME,

        project_root
        / HF_FILENAME,
    ]

    for path in local_paths:

        if path.exists() and path.is_file():

            return path

    # ------------------------------------------------------------
    # Hugging Face
    # ------------------------------------------------------------

    return download_model()


# ================================================================
# EXTRACT STATE DICT
# ================================================================

def _extract_state_dict(
    checkpoint: Any,
) -> Dict[str, torch.Tensor]:

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "MammoSense TB V13 checkpoint is not a dictionary."
        )

    # ------------------------------------------------------------
    # Confirmed V13 structure
    # ------------------------------------------------------------

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        if isinstance(
            state_dict,
            dict,
        ):
            return state_dict

    # ------------------------------------------------------------
    # Common alternatives
    # ------------------------------------------------------------

    for key in (
        "state_dict",
        "model",
        "weights",
    ):

        candidate = checkpoint.get(
            key
        )

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

    # ------------------------------------------------------------
    # Raw state dictionary
    # ------------------------------------------------------------

    if checkpoint:

        if all(
            isinstance(
                key,
                str,
            )
            for key in checkpoint.keys()
        ):

            if any(
                isinstance(
                    value,
                    torch.Tensor,
                )
                for value in checkpoint.values()
            ):
                return checkpoint

    raise RuntimeError(
        "Could not find model_state_dict inside "
        "the MammoSense TB V13 checkpoint."
    )


# ================================================================
# CLEAN STATE DICT
# ================================================================

def _clean_state_dict(
    state_dict,
):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):
            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith(
            "model."
        ):
            new_key = new_key[
                len("model.") :
            ]

        cleaned[
            new_key
        ] = value

    return cleaned


# ================================================================
# LOAD MODEL
# ================================================================

def load_model(
    model_path: Optional[str] = None,
):

    global _MODEL
    global _MODEL_PATH
    global _MODEL_METADATA

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

    checkpoint_path = find_model_path(
        model_path
    )

    # ------------------------------------------------------------
    # Load checkpoint
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
            "MammoSense TB V13 checkpoint could not "
            "be read.\n\n"
            f"Checkpoint: {checkpoint_path}\n\n"
            f"Error: {error}"
        ) from error

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        _MODEL_METADATA = {
            "model_name": checkpoint.get(
                "model_name",
                MODEL_NAME,
            ),
            "version": checkpoint.get(
                "version",
                MODEL_VERSION,
            ),
            "architecture": checkpoint.get(
                "architecture",
                ARCHITECTURE,
            ),
            "image_size": checkpoint.get(
                "image_size",
                IMAGE_SIZE,
            ),
            "input_depth": checkpoint.get(
                "input_depth",
                INPUT_DEPTH,
            ),
            "class_names": checkpoint.get(
                "class_names",
                CLASS_NAMES,
            ),
            "class_to_idx": checkpoint.get(
                "class_to_idx",
                CLASS_TO_IDX,
            ),
            "threshold": checkpoint.get(
                "threshold",
                DEFAULT_THRESHOLD,
            ),
            "dataset": checkpoint.get(
                "dataset",
                "TBX11K",
            ),
            "epochs": checkpoint.get(
                "epochs",
                None,
            ),
            "best_epoch": checkpoint.get(
                "best_epoch",
                None,
            ),
            "best_val_auc": checkpoint.get(
                "best_val_auc",
                None,
            ),
            "best_val_accuracy": checkpoint.get(
                "best_val_accuracy",
                None,
            ),
        }

    else:

        _MODEL_METADATA = {}

    # ------------------------------------------------------------
    # Build EXACT architecture
    # ------------------------------------------------------------

    model = ResNet3D18(
        num_classes=1
    )

    # ------------------------------------------------------------
    # Extract weights
    # ------------------------------------------------------------

    state_dict = _extract_state_dict(
        checkpoint
    )

    state_dict = _clean_state_dict(
        state_dict
    )

    # ------------------------------------------------------------
    # STRICT LOAD
    # ------------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as error:

        expected = set(
            model.state_dict().keys()
        )

        received = set(
            state_dict.keys()
        )

        missing = sorted(
            expected - received
        )

        unexpected = sorted(
            received - expected
        )

        raise RuntimeError(
            "MammoSense TB V13 checkpoint does not "
            "match the expected 3D ResNet-18 architecture.\n\n"

            f"Checkpoint:\n{checkpoint_path}\n\n"

            f"Missing keys: {len(missing)}\n"
            + "\n".join(
                f"  {key}"
                for key in missing[:25]
            )

            + "\n\n"

            f"Unexpected keys: {len(unexpected)}\n"
            + "\n".join(
                f"  {key}"
                for key in unexpected[:25]
            )

            + "\n\n"

            f"Original PyTorch error:\n{error}"
        ) from error

    # ------------------------------------------------------------
    # Device
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
# IMAGE PREPROCESSING
# ================================================================

def preprocess_image(
    image,
) -> torch.Tensor:

    # ------------------------------------------------------------
    # PIL
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
                "Could not open the supplied chest X-ray."
            ) from error

    # ------------------------------------------------------------
    # Grayscale
    # ------------------------------------------------------------

    image = image.convert(
        "L"
    )

    # ------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        Image.Resampling.BILINEAR,
    )

    # ------------------------------------------------------------
    # NumPy
    # ------------------------------------------------------------

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # 0-255 -> 0-1
    # ------------------------------------------------------------

    array /= 255.0

    # ------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------

    array = (
        array - 0.485
    ) / 0.229

    # ------------------------------------------------------------
    # [H,W]
    # -> [1,H,W]
    # ------------------------------------------------------------

    tensor = torch.from_numpy(
        array
    ).unsqueeze(
        0
    )

    # ------------------------------------------------------------
    # [1,H,W]
    # -> [1,1,H,W]
    # ------------------------------------------------------------

    tensor = tensor.unsqueeze(
        1
    )

    # ------------------------------------------------------------
    # Replicate depth
    #
    # [1,1,H,W]
    # -> [1,16,H,W]
    # ------------------------------------------------------------

    tensor = tensor.repeat(
        1,
        INPUT_DEPTH,
        1,
        1,
    )

    # ------------------------------------------------------------
    # [1,16,H,W]
    # -> [1,1,16,H,W]
    # ------------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    return tensor.float()


# ================================================================
# PREDICTION
# ================================================================

def predict(
    image,
    threshold: Optional[float] = None,
    model=None,
):

    if model is None:

        model = load_model()

    # ------------------------------------------------------------
    # Threshold from checkpoint
    # ------------------------------------------------------------

    if threshold is None:

        threshold = DEFAULT_THRESHOLD

        if _MODEL_METADATA:

            threshold = _MODEL_METADATA.get(
                "threshold",
                DEFAULT_THRESHOLD,
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

    with torch.no_grad():

        logits = model(
            tensor
        )

        logits = logits.view(
            -1
        )

        probability = torch.sigmoid(
            logits
        ).item()

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    predicted_label = int(
        probability >= threshold
    )

    predicted_class = (
        "TB"
        if predicted_label == 1
        else "NORMAL"
    )

    confidence = max(
        probability,
        1.0 - probability,
    )

    # ------------------------------------------------------------
    # Clinical-style interpretation
    #
    # This is deliberately screening language.
    # It does NOT claim diagnosis.
    # ------------------------------------------------------------

    if predicted_class == "TB":

        interpretation = (
            "AI screening result suggests "
            "features associated with tuberculosis. "
            "Professional clinical evaluation and "
            "appropriate confirmatory testing are recommended."
        )

    else:

        interpretation = (
            "AI screening result does not indicate "
            "features strongly associated with tuberculosis. "
            "This does not exclude tuberculosis or other disease."
        )

    return {
        "prediction": predicted_class,

        "predicted_class": predicted_class,

        "predicted_label": predicted_label,

        "tb_probability": float(
            probability
        ),

        "normal_probability": float(
            1.0 - probability
        ),

        "non_tb_probability": float(
            1.0 - probability
        ),

        "confidence": float(
            confidence
        ),

        "threshold": float(
            threshold
        ),

        "class_names": CLASS_NAMES,

        "class_to_idx": CLASS_TO_IDX,

        "model_name": MODEL_NAME,

        "model_version": MODEL_VERSION,

        "architecture": ARCHITECTURE,

        "dataset": (
            _MODEL_METADATA.get(
                "dataset",
                "TBX11K",
            )
            if _MODEL_METADATA
            else "TBX11K"
        ),

        "input_shape": [
            1,
            INPUT_CHANNELS,
            INPUT_DEPTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],

        "device": str(
            DEVICE
        ),

        "checkpoint": str(
            _MODEL_PATH
            if _MODEL_PATH
            else ""
        ),

        "interpretation": interpretation,
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
    threshold: Optional[float] = None,
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

    metadata = (
        _MODEL_METADATA
        if _MODEL_METADATA
        else {}
    )

    return {
        "model_name": metadata.get(
            "model_name",
            MODEL_NAME,
        ),

        "version": metadata.get(
            "version",
            MODEL_VERSION,
        ),

        "architecture": metadata.get(
            "architecture",
            ARCHITECTURE,
        ),

        "dataset": metadata.get(
            "dataset",
            "TBX11K",
        ),

        "classes": metadata.get(
            "class_names",
            CLASS_NAMES,
        ),

        "class_to_idx": metadata.get(
            "class_to_idx",
            CLASS_TO_IDX,
        ),

        "image_size": metadata.get(
            "image_size",
            IMAGE_SIZE,
        ),

        "input_depth": metadata.get(
            "input_depth",
            INPUT_DEPTH,
        ),

        "input_channels": INPUT_CHANNELS,

        "pseudo_3d": True,

        "threshold": metadata.get(
            "threshold",
            DEFAULT_THRESHOLD,
        ),

        "device": str(
            DEVICE
        ),

        "checkpoint": str(
            _MODEL_PATH
            if _MODEL_PATH
            else ""
        ),

        "epochs": metadata.get(
            "epochs"
        ),

        "best_epoch": metadata.get(
            "best_epoch"
        ),

        "best_val_auc": metadata.get(
            "best_val_auc"
        ),

        "best_val_accuracy": metadata.get(
            "best_val_accuracy"
        ),
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

    dummy = torch.zeros(
        (
            1,
            1,
            INPUT_DEPTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():

        output = model(
            dummy
        )

    if tuple(
        output.shape
    ) != (1, 1):

        raise RuntimeError(
            "MammoSense TB V13 produced "
            f"unexpected output shape: "
            f"{tuple(output.shape)}"
        )

    probability = torch.sigmoid(
        output
    ).item()

    return {
        "loaded": True,

        "model": MODEL_NAME,

        "checkpoint": str(
            _MODEL_PATH
        ),

        "device": str(
            DEVICE
        ),

        "input_shape": [
            1,
            1,
            INPUT_DEPTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],

        "output_shape": list(
            output.shape
        ),

        "sample_probability": float(
            probability
        ),
    }


# ================================================================
# DIRECT TEST
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAMMOSENSE TB V13")
    print("Custom 3D ResNet-18")
    print("=" * 70)

    result = verify_model()

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print("=" * 70)
    print("MODEL READY")
    print("=" * 70)
