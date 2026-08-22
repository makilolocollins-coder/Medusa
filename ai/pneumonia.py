# ================================================================
# MAMMOSENSE PNEUMONIA V2
# 3D RESNET-18 PSEUDO-3D CHEST X-RAY CLASSIFIER
#
# INPUT TO MODEL:
#   [B, C, D, H, W]
#   [1, 1, 16, 224, 224]
#
# CLASSES:
#   0 = NORMAL
#   1 = PNEUMONIA
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
# THIS MUST MATCH TRAINING ARCHITECTURE.
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
                stride=stride,
            )
        ]

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

    def forward(self, x):

        # --------------------------------------------------------
        # DEFENSIVE 4D -> 5D CONVERSION
        #
        # [B,C,H,W]
        # ->
        # [B,C,1,H,W]
        # ->
        # [B,C,16,H,W]
        # --------------------------------------------------------

        if x.ndim == 4:

            x = x.unsqueeze(2)

            x = x.repeat(
                1,
                1,
                DEPTH,
                1,
                1,
            )

        # --------------------------------------------------------
        # REQUIRE 5D
        # --------------------------------------------------------

        if x.ndim != 5:

            raise RuntimeError(
                "MammoSense Pneumonia requires "
                "[B,C,D,H,W]. "
                f"Received {tuple(x.shape)}"
            )

        # --------------------------------------------------------
        # CHANNEL
        # --------------------------------------------------------

        if x.shape[1] != 1:

            raise RuntimeError(
                "MammoSense Pneumonia requires "
                "one input channel. "
                f"Received {x.shape[1]}"
            )

        # --------------------------------------------------------
        # DEPTH
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
                    "Invalid pseudo-3D depth. "
                    f"Expected {DEPTH}, "
                    f"received {x.shape[2]}"
                )

        # --------------------------------------------------------
        # NETWORK
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
# ALIAS
#
# Useful if detection.py imports PneumoniaModel.
# ================================================================

PneumoniaModel = ResNet3D18


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

    if _model is not None:
        return _model

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
    )

    model = ResNet3D18(
        num_classes=2
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    # ------------------------------------------------------------
    # EXTRACT STATE DICT
    # ------------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

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

        if new_key.startswith("module."):
            new_key = new_key[7:]

        if new_key.startswith("model."):
            new_key = new_key[6:]

        cleaned_state_dict[
            new_key
        ] = value

    # ------------------------------------------------------------
    # VERIFY CHECKPOINT
    # ------------------------------------------------------------

    expected_keys = set(
        model.state_dict().keys()
    )

    actual_keys = set(
        cleaned_state_dict.keys()
    )

    missing = sorted(
        expected_keys - actual_keys
    )

    unexpected = sorted(
        actual_keys - expected_keys
    )

    if missing:

        raise RuntimeError(
            "Pneumonia checkpoint mismatch.\n\n"
            "Missing keys:\n"
            + "\n".join(missing[:30])
        )

    if unexpected:

        raise RuntimeError(
            "Pneumonia checkpoint mismatch.\n\n"
            "Unexpected keys:\n"
            + "\n".join(unexpected[:30])
        )

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

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
# PREDICT
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
    # GRAYSCALE
    # ------------------------------------------------------------

    image = image.convert(
        "L"
    )

    # ------------------------------------------------------------
    # 2D TRANSFORM
    #
    # [1,224,224]
    # ------------------------------------------------------------

    tensor = transform(
        image
    )

    if tensor.ndim != 3:

        raise RuntimeError(
            "Invalid preprocessing output. "
            f"Received {tuple(tensor.shape)}"
        )

    if tensor.shape != (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):

        raise RuntimeError(
            "Invalid X-ray size. "
            f"Expected [1,224,224], "
            f"received {tuple(tensor.shape)}"
        )

    # ------------------------------------------------------------
    # CREATE DEPTH
    #
    # [1,224,224]
    # ->
    # [1,16,224,224]
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

    # ------------------------------------------------------------
    # ADD BATCH
    #
    # [1,16,224,224]
    # ->
    # [1,1,16,224,224]
    # ------------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    # ------------------------------------------------------------
    # HARD CHECK
    # ------------------------------------------------------------

    expected = (
        1,
        1,
        DEPTH,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tensor.shape != expected:

        raise RuntimeError(
            "Pseudo-3D construction failed. "
            f"Expected {expected}, "
            f"received {tuple(tensor.shape)}"
        )

    # ------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------

    tensor = tensor.to(
        DEVICE,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------
    # FINAL 5D CHECK
    # ------------------------------------------------------------

    if tensor.ndim != 5:

        raise RuntimeError(
            "FATAL: Pneumonia input is not 5D. "
            f"Received {tensor.ndim}D."
        )

    # ------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------

    logits = model(
        tensor
    )

    # ------------------------------------------------------------
    # PROBABILITIES
    # ------------------------------------------------------------

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

    probability_dict = {
        "NORMAL": normal_probability,
        "PNEUMONIA": pneumonia_probability,
    }

    # ------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # MEDUSA RESULT FORMAT
    # ------------------------------------------------------------

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_dict,
    }


# ================================================================
# OPTIONAL COMPATIBILITY ALIASES
# ================================================================
#
# These allow detection.py to use either common naming style.
# ================================================================

predict_pneumonia = predict
get_prediction = predict
