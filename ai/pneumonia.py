# ================================================================
# MAMMOSENSE PNEUMONIA V2
# 3D RESNET-18 PSEUDO-3D CHEST X-RAY CLASSIFIER
# ================================================================

import os
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
# 3D RESNET BASIC BLOCK
# ================================================================

class BasicBlock3D(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
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

        self.downsample = downsample

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ================================================================
# 3D RESNET-18
# ================================================================

class ResNet3D18(nn.Module):

    def __init__(
        self,
        num_classes=2,
        input_channels=1,
    ):

        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv3d(
            input_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(64)

        self.relu = nn.ReLU(
            inplace=True
        )

        self.maxpool = nn.MaxPool3d(
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=1,
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
        channels,
        blocks,
        stride,
    ):

        downsample = None

        if (
            stride != 1
            or self.in_channels != channels
        ):

            downsample = nn.Sequential(

                nn.Conv3d(
                    self.in_channels,
                    channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),

                nn.BatchNorm3d(
                    channels
                ),
            )

        layers = []

        layers.append(
            BasicBlock3D(
                self.in_channels,
                channels,
                stride,
                downsample,
            )
        )

        self.in_channels = channels

        for _ in range(1, blocks):

            layers.append(
                BasicBlock3D(
                    self.in_channels,
                    channels,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

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
# MODEL LOADING
# ================================================================

_model = None


def load_model():

    global _model

    if _model is not None:
        return _model

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
    )

    model = ResNet3D18(
        num_classes=2,
        input_channels=1,
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # ------------------------------------------------------------
    # EXTRACT STATE DICT
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # CLEAN COMMON PREFIXES
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
    # LOAD WEIGHTS
    # ------------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(DEVICE)

    model.eval()

    _model = model

    return _model


# ================================================================
# PREPROCESSING
# ================================================================

transform = transforms.Compose([

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
])


# ================================================================
# PREDICTION
# ================================================================

@torch.no_grad()
def predict(image):

    model = load_model()

    if not isinstance(
        image,
        Image.Image,
    ):

        image = Image.open(image)

    image = image.convert(
        "L"
    )

    tensor = transform(
        image
    )

    # ------------------------------------------------------------
    # 2D IMAGE → PSEUDO 3D VOLUME
    #
    # Current shape:
    # 1 × 224 × 224
    #
    # Required:
    # 1 × 16 × 224 × 224
    # ------------------------------------------------------------

    tensor = tensor.unsqueeze(
        1
    )

    tensor = tensor.repeat(
        1,
        DEPTH,
        1,
        1,
    )

    # Add batch dimension
    #
    # 1 × 16 × 224 × 224
    # →
    # 1 × 1 × 16 × 224 × 224

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
