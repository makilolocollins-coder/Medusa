# ================================================================
# MAMMOSENSE PNEUMONIA V2
# 3D RESNET-18 PSEUDO-3D CHEST X-RAY CLASSIFIER
#
# MUST MATCH THE TRAINING ARCHITECTURE EXACTLY
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

_model = None


# ================================================================
# BASIC BLOCK
# EXACTLY MATCHES TRAINING CODE
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

        self.relu = nn.ReLU(
            inplace=True
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
# RESNET-18
#
# THIS MUST MATCH TRAINING EXACTLY
# ================================================================

class ResNet3D18(nn.Module):

    def __init__(
        self,
        num_classes=2,
    ):

        super().__init__()

        self.in_channels = 64

        # IMPORTANT:
        # Training used "stem", not conv1/bn1/maxpool
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

        self.avgpool = nn.AdaptiveAvgPool3d(
            (1, 1, 1)
        )

        self.fc = nn.Linear(
            512,
            num_classes,
        )

    def _make_layer(
        self,
        out_channels,
        blocks,
        stride,
    ):

        layers = [

            BasicBlock3D(
                self.in_channels,
                out_channels,
                stride,
            )

        ]

        self.in_channels = out_channels

        for _ in range(1, blocks):

            layers.append(

                BasicBlock3D(
                    out_channels,
                    out_channels,
                )

            )

        return nn.Sequential(
            *layers
        )

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
# PREPROCESSING
#
# MUST MATCH VALIDATION/TEST PREPROCESSING
# ================================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485],
        std=[0.229],
    ),
])


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    global _model

    if _model is not None:

        return _model

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # ============================================================
    # CREATE EXACT TRAINING ARCHITECTURE
    # ============================================================

    model = ResNet3D18(
        num_classes=2
    )

    # ============================================================
    # EXTRACT STATE DICT
    # ============================================================

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

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # ============================================================
    # REMOVE COMMON PREFIXES
    # ============================================================

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

    # ============================================================
    # STRICT LOAD
    # ============================================================

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    _model = model

    return _model


# ================================================================
# PREDICTION
# ================================================================

@torch.no_grad()
def predict(image):

    model = load_model()

    # ------------------------------------------------------------
    # ACCEPT PIL IMAGE OR FILE PATH
    # ------------------------------------------------------------

    if not isinstance(
        image,
        Image.Image,
    ):

        image = Image.open(
            image
        )

    # ------------------------------------------------------------
    # CONVERT TO GRAYSCALE
    # ------------------------------------------------------------

    image = image.convert(
        "L"
    )

    # ------------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------------

    tensor = transform(
        image
    )

    # Current:
    #
    # [1, 224, 224]
    #
    # Repeat across depth:
    #
    # [16, 224, 224]
    #
    # Add channel:
    #
    # [1, 16, 224, 224]
    #
    # Add batch:
    #
    # [1, 1, 16, 224, 224]

    tensor = tensor.repeat(
        DEPTH,
        1,
        1,
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    # ------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    normal_probability = float(
        probabilities[0].item()
    )

    pneumonia_probability = float(
        probabilities[1].item()
    )

    probabilities_dict = {

        "NORMAL":
            normal_probability,

        "PNEUMONIA":
            pneumonia_probability,

    }

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

    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probabilities_dict,

    }
