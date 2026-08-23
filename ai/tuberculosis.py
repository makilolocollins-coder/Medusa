# ================================================================
# MAMMOSENSE TB V12
# TBX11K | PSEUDO-3D RESNET-18
#
# IMPORTANT:
# This architecture MUST remain identical to the architecture
# used when mammosense_tb_v12.pt was trained.
#
# Input:
#   Grayscale chest X-ray
#   -> 224 x 224
#   -> normalized
#   -> replicated 16 times
#   -> [1, 16, 224, 224]
#
# Output:
#   NON_TB = 0
#   TB     = 1
# ================================================================

from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_NAME = "MammoSense TB V12"

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
# THIS IS COPIED STRUCTURALLY FROM THE TRAINING CODE.
#
# Do NOT replace:
#
#     self.conv1
#     self.bn1
#
# with:
#
#     self.conv1 = nn.Sequential(...)
#
# because that changes the checkpoint key names.
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

        # --------------------------------------------------------
        # Conv 1
        # --------------------------------------------------------

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        # --------------------------------------------------------
        # BatchNorm 1
        # --------------------------------------------------------

        self.bn1 = nn.BatchNorm3d(
            out_channels
        )

        # --------------------------------------------------------
        # Conv 2
        # --------------------------------------------------------

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # --------------------------------------------------------
        # BatchNorm 2
        # --------------------------------------------------------

        self.bn2 = nn.BatchNorm3d(
            out_channels
        )

        # --------------------------------------------------------
        # Shortcut
        # --------------------------------------------------------

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

    # ============================================================
    # FORWARD
    # ============================================================

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

        out += identity

        out = F.relu(
            out,
            inplace=True,
        )

        return out


# ================================================================
# RESNET-18 3D
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

            nn.BatchNorm3d(64),

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
            64,
            2,
            stride=1,
        )

        # --------------------------------------------------------
        # LAYER 2
        # --------------------------------------------------------

        self.layer2 = self._make_layer(
            128,
            2,
            stride=2,
        )

        # --------------------------------------------------------
        # LAYER 3
        # --------------------------------------------------------

        self.layer3 = self._make_layer(
            256,
            2,
            stride=2,
        )

        # --------------------------------------------------------
        # LAYER 4
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # INITIALIZATION
        # --------------------------------------------------------

        self._initialize()

    # ============================================================
    # MAKE RESNET LAYER
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
# CHECKPOINT PATH DISCOVERY
# ================================================================

def _possible_model_paths():

    current_file = Path(__file__).resolve()

    project_root = current_file.parent.parent

    paths = [

        # --------------------------------------------------------
        # Recommended project location
        # --------------------------------------------------------

        project_root
        / "models"
        / "mammosense_tb_v12.pt",

        project_root
        / "model"
        / "mammosense_tb_v12.pt",

        # --------------------------------------------------------
        # AI directory
        # --------------------------------------------------------

        current_file.parent
        / "mammosense_tb_v12.pt",

        current_file.parent
        / "mammosense_tb_v12"
        / "mammosense_tb_v12.pt",

        # --------------------------------------------------------
        # Root
        # --------------------------------------------------------

        project_root
        / "mammosense_tb_v12.pt",

        # --------------------------------------------------------
        # Alternative names used during development
        # --------------------------------------------------------

        project_root
        / "models"
        / "mammosense_tb_v12 (1).pt",

        project_root
        / "models"
        / "mammosense_tb_v12 (2).pt",

        # --------------------------------------------------------
        # Hugging Face / downloaded location
        # --------------------------------------------------------

        Path("/tmp/mammosense_tb_v12.pt"),

        Path("/app/mammosense_tb_v12.pt"),

        Path("/mount/src/medusa/mammosense_tb_v12.pt"),

        Path("/mount/src/medusa/models/mammosense_tb_v12.pt"),
    ]

    # Remove duplicates while preserving order.

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
# FIND MODEL
# ================================================================

def find_model_path(
    explicit_path: Optional[str] = None,
) -> Path:

    # ------------------------------------------------------------
    # Explicit path
    # ------------------------------------------------------------

    if explicit_path:

        path = Path(explicit_path)

        if path.exists():

            return path

        raise FileNotFoundError(
            f"MammoSense TB model was not found at:\n{path}"
        )

    # ------------------------------------------------------------
    # Search standard locations
    # ------------------------------------------------------------

    for path in _possible_model_paths():

        if path.exists() and path.is_file():

            return path

    # ------------------------------------------------------------
    # Nothing found
    # ------------------------------------------------------------

    searched = "\n".join(
        f"  - {p}"
        for p in _possible_model_paths()
    )

    raise FileNotFoundError(
        "MammoSense TB V12 checkpoint could not be found.\n\n"
        "Searched:\n"
        f"{searched}"
    )


# ================================================================
# CHECKPOINT STATE DICT EXTRACTION
# ================================================================

def _extract_state_dict(
    checkpoint: Any,
) -> Dict[str, torch.Tensor]:

    # ------------------------------------------------------------
    # Standard V12 checkpoint
    # ------------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

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
        # Other common checkpoint names
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

                    return candidate

        # --------------------------------------------------------
        # Raw state dict
        # --------------------------------------------------------

        if all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):

            if any(
                isinstance(
                    v,
                    torch.Tensor,
                )
                for v in checkpoint.values()
            ):

                return checkpoint

    raise RuntimeError(
        "Could not locate a valid model state_dict "
        "inside the MammoSense TB checkpoint."
    )


# ================================================================
# REMOVE DATA PARALLEL PREFIX
# ================================================================

def _clean_state_dict(
    state_dict: Dict[str, torch.Tensor],
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

        cleaned[new_key] = value

    return cleaned


# ================================================================
# MODEL LOADING
# ================================================================

_MODEL = None

_MODEL_PATH = None


def load_model(
    model_path: Optional[str] = None,
):

    global _MODEL
    global _MODEL_PATH

    # ------------------------------------------------------------
    # Return cached model
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
    # Build EXACT V12 architecture
    # ------------------------------------------------------------

    model = ResNet3D18(
        num_classes=1
    )

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        # Compatibility with older PyTorch.

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    except Exception as error:

        raise RuntimeError(
            "MammoSense TB V12 checkpoint could not be "
            "read.\n\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Error: {error}"
        ) from error

    # ------------------------------------------------------------
    # Extract weights
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
            "MammoSense TB V12 checkpoint does not contain "
            "a valid model state dictionary.\n\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Error: {error}"
        ) from error

    # ------------------------------------------------------------
    # Strict loading
    #
    # IMPORTANT:
    # We intentionally use strict=True.
    #
    # If this fails, it means the checkpoint is genuinely not
    # the V12 architecture and should NOT silently be accepted.
    # ------------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as error:

        # --------------------------------------------------------
        # Generate useful diagnostic information
        # --------------------------------------------------------

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
            "MammoSense TB V12 checkpoint could not be "
            "loaded into the expected 3D ResNet-18 architecture.\n\n"

            f"Checkpoint: {checkpoint_path}\n\n"

            f"Missing keys ({len(missing)}):\n"
            + "\n".join(
                f"  {key}"
                for key in missing[:30]
            )

            + (
                "\n  ..."
                if len(missing) > 30
                else ""
            )

            + f"\n\nUnexpected keys ({len(unexpected)}):\n"

            + "\n".join(
                f"  {key}"
                for key in unexpected[:30]
            )

            + (
                "\n  ..."
                if len(unexpected) > 30
                else ""
            )

            + "\n\n"
            + f"Original PyTorch error:\n{error}"
        ) from error

    # ------------------------------------------------------------
    # Move to device
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
    # PIL conversion
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
                "Could not open the supplied chest X-ray image."
            ) from error

    # ------------------------------------------------------------
    # Grayscale
    #
    # EXACTLY like training:
    #
    # Image.open(path).convert("L")
    # ------------------------------------------------------------

    image = image.convert("L")

    # ------------------------------------------------------------
    # Resize
    #
    # EXACTLY:
    #
    # 224 x 224
    # ------------------------------------------------------------

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        Image.Resampling.BILINEAR,
    )

    # ------------------------------------------------------------
    # NumPy conversion
    #
    # Avoiding a torchvision normalization dependency keeps
    # inference stable.
    # ------------------------------------------------------------

    import numpy as np

    arr = np.asarray(
        image,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # 0-255 -> 0-1
    # ------------------------------------------------------------

    arr /= 255.0

    # ------------------------------------------------------------
    # EXACT TRAINING NORMALIZATION
    #
    # (arr - 0.485) / 0.229
    # ------------------------------------------------------------

    arr = (
        arr - 0.485
    ) / 0.229

    # ------------------------------------------------------------
    # [H,W]
    # ->
    # [1,H,W]
    # ------------------------------------------------------------

    tensor = torch.from_numpy(
        arr
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
    # Replicate 16 times
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
# PREDICTION
# ================================================================

def predict(
    image,
    threshold: float = DEFAULT_THRESHOLD,
    model=None,
):

    if model is None:

        model = load_model()

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

    # ------------------------------------------------------------
    # Return rich result
    # ------------------------------------------------------------

    return {
        "prediction": predicted_class,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "tb_probability": float(
            probability
        ),
        "non_tb_probability": float(
            1.0 - probability
        ),
        "confidence": float(
            max(
                probability,
                1.0 - probability,
            )
        ),
        "threshold": float(
            threshold
        ),
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "model_name": MODEL_NAME,
        "architecture": "3D ResNet-18",
        "input_shape": [
            1,
            DEPTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],
        "device": str(
            DEVICE
        ),
    }


# ================================================================
# COMPATIBILITY FUNCTION
# ================================================================
#
# detection.py currently calls:
#
#     load_tb_model()
#
# Therefore this function MUST exist.
# ================================================================

def load_tb_model(
    model_path: Optional[str] = None,
):

    return load_model(
        model_path=model_path
    )


# ================================================================
# COMPATIBILITY PREDICTION FUNCTION
# ================================================================

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
        "model_name": MODEL_NAME,
        "version": "12.0",
        "architecture": "3D ResNet-18",
        "dataset": "TBX11K",
        "task": "Binary tuberculosis classification",
        "classes": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "input_channels": INPUT_CHANNELS,
        "image_size": IMAGE_SIZE,
        "depth": DEPTH,
        "pseudo_3d": True,
        "input_shape": [
            INPUT_CHANNELS,
            DEPTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],
        "normalization_mean": 0.485,
        "normalization_std": 0.229,
        "threshold": DEFAULT_THRESHOLD,
        "device": str(DEVICE),
    }


# ================================================================
# STARTUP VERIFICATION
# ================================================================

def verify_model(
    model_path: Optional[str] = None,
):

    model = load_model(
        model_path=model_path
    )

    # ------------------------------------------------------------
    # Dummy tensor with EXACT expected input
    # ------------------------------------------------------------

    dummy = torch.zeros(
        (
            1,
            1,
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

    with torch.no_grad():

        output = model(
            dummy
        )

    if tuple(
        output.shape
    ) != (1, 1):

        raise RuntimeError(
            "MammoSense TB V12 model loaded, but produced "
            f"an unexpected output shape: {tuple(output.shape)}"
        )

    probability = torch.sigmoid(
        output
    ).item()

    return {
        "loaded": True,
        "checkpoint": str(
            _MODEL_PATH
        ),
        "device": str(
            DEVICE
        ),
        "input_shape": [
            1,
            1,
            DEPTH,
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
# OPTIONAL STARTUP TEST
# ================================================================

if __name__ == "__main__":

    print("=" * 64)

    print(
        "MAMMOSENSE TB V12"
    )

    print(
        "3D ResNet-18 checkpoint loader"
    )

    print("=" * 64)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Input:",
        (
            1,
            1,
            DEPTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
    )

    print(
        "\nSearching for checkpoint..."
    )

    try:

        result = verify_model()

        print(
            "\n✓ MODEL LOADED"
        )

        print(
            "Checkpoint:",
            result["checkpoint"],
        )

        print(
            "Output shape:",
            result["output_shape"],
        )

        print(
            "Sample probability:",
            f"{result['sample_probability']:.6f}",
        )

    except Exception as error:

        print(
            "\n✗ MODEL VERIFICATION FAILED"
        )

        print(
            error
        )
