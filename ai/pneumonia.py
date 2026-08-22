# ================================================================
# MAMMOSENSE PNEUMONIA V2
# 3D RESNET-18 PSEUDO-3D CHEST X-RAY CLASSIFIER
#
# INPUT:
#   [B, C, D, H, W]
#   [1, 1, 16, 224, 224]
#
# CLASSES:
#   0 = NORMAL
#   1 = PNEUMONIA
#
# CHECKPOINT:
#   Makky07/Mammosense_pneumonia
#   mammosense_pneumonia_v2.pt
# ================================================================

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download


# ================================================================
# CONFIGURATION
# ================================================================

REPO_ID = "Makky07/Mammosense_pneumonia"

MODEL_FILENAME = "mammosense_pneumonia_v2.pt"

IMAGE_SIZE = 224
DEPTH = 16

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# 3D BASIC BLOCK
# ================================================================

class BasicBlock3D(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
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

        self.relu = nn.ReLU(
            inplace=True
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

            self.downsample = nn.Sequential(

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

            self.downsample = None

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        out = self.relu(out)

        return out


# ================================================================
# 3D RESNET-18
#
# IMPORTANT:
# This architecture exactly matches the architecture used during
# training.
#
# The trained checkpoint contains:
#
#   stem.0.weight
#   stem.1.weight
#   ...
#
# Therefore we MUST use self.stem here.
# ================================================================

class ResNet3D18(nn.Module):

    def __init__(
        self,
        num_classes=2,
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
                stride=2,
                padding=1,
            ),
        )

        # --------------------------------------------------------
        # RESNET LAYERS
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
        # CLASSIFIER
        # --------------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool3d(
            (1, 1, 1)
        )

        self.fc = nn.Linear(
            512,
            num_classes,
        )

    # ============================================================
    # MAKE RESNET LAYER
    # ============================================================

    def _make_layer(
        self,
        out_channels,
        blocks,
        stride,
    ):

        layers = []

        layers.append(
            BasicBlock3D(
                self.in_channels,
                out_channels,
                stride,
            )
        )

        self.in_channels = out_channels

        for _ in range(
            1,
            blocks,
        ):

            layers.append(
                BasicBlock3D(
                    out_channels,
                    out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(
            *layers
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

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1,
        )

        x = self.fc(x)

        return x


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485],
        std=[0.229],
    ),
])


# ================================================================
# GLOBAL MODEL CACHE
# ================================================================

_model = None


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    global _model

    # ------------------------------------------------------------
    # Return cached model
    # ------------------------------------------------------------

    if _model is not None:
        return _model

    # ------------------------------------------------------------
    # Download checkpoint
    # ------------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
    )

    # ------------------------------------------------------------
    # Create EXACT training architecture
    # ------------------------------------------------------------

    model = ResNet3D18(
        num_classes=2,
    )

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # ------------------------------------------------------------
    # Extract model weights
    # ------------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # ------------------------------------------------------------
    # Remove common training prefixes
    # ------------------------------------------------------------

    cleaned_state_dict = {}

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

        cleaned_state_dict[
            new_key
        ] = value

    # ------------------------------------------------------------
    # Verify architecture before loading
    # ------------------------------------------------------------

    expected_keys = set(
        model.state_dict().keys()
    )

    checkpoint_keys = set(
        cleaned_state_dict.keys()
    )

    missing_keys = (
        expected_keys
        - checkpoint_keys
    )

    unexpected_keys = (
        checkpoint_keys
        - expected_keys
    )

    if missing_keys:

        raise RuntimeError(
            "MammoSense Pneumonia checkpoint "
            "does not match the deployed model "
            "architecture.\n\n"
            f"Missing keys:\n"
            f"{sorted(missing_keys)[:20]}"
        )

    if unexpected_keys:

        raise RuntimeError(
            "MammoSense Pneumonia checkpoint "
            "contains unexpected parameters.\n\n"
            f"Unexpected keys:\n"
            f"{sorted(unexpected_keys)[:20]}"
        )

    # ------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    model.to(
        DEVICE
    )

    # ------------------------------------------------------------
    # Evaluation mode
    # ------------------------------------------------------------

    model.eval()

    # ------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------

    _model = model

    return _model


# ================================================================
# PREDICTION
# ================================================================

@torch.no_grad()
def predict(image):

    model = load_model()

    # ============================================================
    # 1. LOAD IMAGE
    # ============================================================

    if not isinstance(
        image,
        Image.Image,
    ):

        image = Image.open(
            image
        )

    # ============================================================
    # 2. GRAYSCALE
    # ============================================================

    image = image.convert(
        "L"
    )

    # ============================================================
    # 3. 2D PREPROCESSING
    #
    # Result:
    #
    # [C, H, W]
    #
    # [1, 224, 224]
    # ============================================================

    tensor = transform(
        image
    )

    if tensor.ndim != 3:

        raise RuntimeError(
            "Pneumonia preprocessing error: "
            f"expected 3D tensor [C,H,W], "
            f"got {tuple(tensor.shape)}"
        )

    if tensor.shape != (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):

        raise RuntimeError(
            "Pneumonia preprocessing error: "
            f"expected [1,224,224], "
            f"got {tuple(tensor.shape)}"
        )

    # ============================================================
    # 4. CREATE PSEUDO-3D VOLUME
    #
    # Current:
    #
    # [1, 224, 224]
    #
    # Add depth:
    #
    # [1, 1, 224, 224]
    #
    # Repeat depth:
    #
    # [1, 16, 224, 224]
    #
    # This represents:
    #
    # [C, D, H, W]
    # ============================================================

    tensor = tensor.unsqueeze(
        1
    )

    tensor = tensor.repeat(
        1,
        DEPTH,
        1,
        1,
    )

    if tensor.shape != (
        1,
        DEPTH,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):

        raise RuntimeError(
            "Pseudo-3D construction failed: "
            f"expected [1,16,224,224], "
            f"got {tuple(tensor.shape)}"
        )

    # ============================================================
    # 5. ADD BATCH DIMENSION
    #
    # Current:
    #
    # [C, D, H, W]
    #
    # Add batch:
    #
    # [B, C, D, H, W]
    #
    # FINAL:
    #
    # [1, 1, 16, 224, 224]
    # ============================================================

    tensor = tensor.unsqueeze(
        0
    )

    # ============================================================
    # 6. HARD 5D VALIDATION
    # ============================================================

    if tensor.ndim != 5:

        raise RuntimeError(
            "FATAL PNEUMONIA INPUT ERROR: "
            f"3D ResNet requires a 5D tensor "
            f"[B,C,D,H,W], but received "
            f"{tensor.ndim}D: "
            f"{tuple(tensor.shape)}"
        )

    expected_shape = (
        1,
        1,
        DEPTH,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tensor.shape != expected_shape:

        raise RuntimeError(
            "FATAL PNEUMONIA SHAPE ERROR: "
            f"expected {expected_shape}, "
            f"received {tuple(tensor.shape)}"
        )

    # ============================================================
    # 7. MOVE TO DEVICE
    # ============================================================

    tensor = tensor.to(
        DEVICE,
        dtype=torch.float32,
    )

    # ============================================================
    # 8. FINAL DEVICE/SHAPE CHECK
    # ============================================================

    if tensor.ndim != 5:

        raise RuntimeError(
            "Pneumonia tensor is not 5D "
            "immediately before inference."
        )

    # ============================================================
    # 9. MODEL INFERENCE
    # ============================================================

    logits = model(
        tensor
    )

    # ============================================================
    # 10. SOFTMAX
    # ============================================================

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    # ============================================================
    # 11. CLASS PROBABILITIES
    # ============================================================

    normal_probability = float(
        probabilities[0].item()
    )

    pneumonia_probability = float(
        probabilities[1].item()
    )

    probability_dict = {

        "NORMAL":
            normal_probability,

        "PNEUMONIA":
            pneumonia_probability,
    }

    # ============================================================
    # 12. PREDICTION
    # ============================================================

    predicted_index = int(
        torch.argmax(
            probabilities
        ).item()
    )

    prediction = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )

    # ============================================================
    # 13. RETURN STANDARD MEDUSА FORMAT
    # ============================================================

    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probability_dict,
    }
