# ================================================================
# MAMMOSENSE PNEUMONIA V2
# ROBUST 3D RESNET-18 PSEUDO-3D CHEST X-RAY CLASSIFIER
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
#
# IMPORTANT:
#   This file is designed to prevent the
#   "expected 5D input (got 4D input)" error.
# ================================================================

# ================================================================
# DEBUG VERSION IDENTIFIER
# ================================================================

print("🔥 MAMMOSENSE PNEUMONIA V2 LOADED")
print("🔥 FILE:", __file__)
print("🔥 VERSION: PNEUMONIA_V2_5D_FIX")

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
    # MAKE LAYER
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
                stride=stride,
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
    #
    # ACCEPTS:
    #
    #   [B,C,D,H,W]
    #
    # AND DEFENSIVELY CONVERTS:
    #
    #   [B,C,H,W]
    #
    # TO:
    #
    #   [B,C,16,H,W]
    # ============================================================

    def forward(self, x):

        print(
            "🔥 MODEL FORWARD RECEIVED:",
            tuple(x.shape),
            "ndim=",
            x.ndim,
        )

        # --------------------------------------------------------
        # 4D -> 5D
        # --------------------------------------------------------

        if x.ndim == 4:

            print(
                "⚠️ 4D INPUT DETECTED INSIDE MODEL."
                " Converting to pseudo-3D."
            )

            x = x.unsqueeze(2)

            x = x.repeat(
                1,
                1,
                DEPTH,
                1,
                1,
            )

            print(
                "🔥 AFTER 4D->5D:",
                tuple(x.shape),
            )

        # --------------------------------------------------------
        # REQUIRE 5D
        # --------------------------------------------------------

        if x.ndim != 5:

            raise RuntimeError(
                "MammoSense Pneumonia requires "
                "5D input [B,C,D,H,W]. "
                f"Received shape: {tuple(x.shape)}"
            )

        # --------------------------------------------------------
        # CHANNEL CHECK
        # --------------------------------------------------------

        if x.shape[1] != 1:

            raise RuntimeError(
                "MammoSense Pneumonia requires "
                "1 input channel. "
                f"Received {x.shape[1]} channels."
            )

        # --------------------------------------------------------
        # DEPTH CHECK
        # --------------------------------------------------------

        if x.shape[2] != DEPTH:

            if x.shape[2] == 1:

                x = x.repeat(
                    1,
                    1,
                    DEPTH,
                    1,
                    1,
                )

            else:

                raise RuntimeError(
                    "Incorrect pseudo-3D depth. "
                    f"Expected {DEPTH}, "
                    f"received {x.shape[2]}."
                )

        print(
            "🔥 MODEL FINAL INPUT:",
            tuple(x.shape),
            "ndim=",
            x.ndim,
        )

        # --------------------------------------------------------
        # RESNET
        # --------------------------------------------------------

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
# IMAGE TRANSFORM
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
# MODEL CACHE
# ================================================================

_model = None


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    global _model

    # ------------------------------------------------------------
    # CACHE
    # ------------------------------------------------------------

    if _model is not None:

        return _model

    print(
        "🔥 Loading MammoSense Pneumonia model..."
    )

    # ------------------------------------------------------------
    # DOWNLOAD
    # ------------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
    )

    print(
        "🔥 Checkpoint:",
        model_path,
    )

    # ------------------------------------------------------------
    # CREATE EXACT TRAINING ARCHITECTURE
    # ------------------------------------------------------------

    model = ResNet3D18(
        num_classes=2
    )

    # ------------------------------------------------------------
    # LOAD CHECKPOINT
    # ------------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    # ------------------------------------------------------------
    # EXTRACT STATE DICT
    # ------------------------------------------------
