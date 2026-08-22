# ================================================================
# MAMMOSENSE PNEUMONIA V2
# 3D RESNET-18 PSEUDO-3D CHEST X-RAY CLASSIFIER
#
# TRAINING ARCHITECTURE:
#   stem
#   layer1
#   layer2
#   layer3
#   layer4
#   avgpool
#   fc
#
# INPUT:
#   [B, C, D, H, W]
#   [1, 1, 16, 224, 224]
#
# CLASSES:
#   0 = NORMAL
#   1 = PNEUMONIA
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
NUM_CLASSES = 2

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA",
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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

        self.relu = nn.ReLU(
            inplace=True
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

            self.downsample = nn.Sequential(

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

            self.downsample = None

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:

            identity = self.downsample(
                identity
            )

        out = out + identity

        out = self.relu(out)

        return out


# ================================================================
# 3D RESNET-18
#
# THIS MATCHES THE TRAINING CODE PROVIDED BY THE USER.
# ================================================================

class ResNet3D18(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
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
                stride=2,
                padding=1,
            ),
        )

        # --------------------------------------------------------
        # RESNET LAYERS
        # --------------------------------------------------------

        self.layer1 = self._make_layer(
            out_channels=64,
            blocks=2,
            stride=1,
        )

        self.layer2 = self._make_layer(
            out_channels=128,
            blocks=2,
            stride=2,
        )

        self.layer3 = self._make_layer(
            out_channels=256,
            blocks=2,
            stride=2,
        )

        self.layer4 = self._make_layer(
            out_channels=512,
            blocks=2,
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

        layers = []

        layers.append(
            BasicBlock3D(
                in_channels=self.in_channels,
                out_channels=out_channels,
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
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(
            *layers
        )

    def forward(self, x):

        # --------------------------------------------------------
        # ABSOLUTE INPUT VALIDATION
        # --------------------------------------------------------

        if x.ndim != 5:

            raise RuntimeError(
                "MammoSense Pneumonia received "
                f"{x.ndim}D input {tuple(x.shape)}. "
                "Expected [B,C,D,H,W], e.g. "
                "[1,1,16,224,224]."
            )

        if x.shape[1] != 1:

            raise RuntimeError(
                "MammoSense Pneumonia expects "
                f"1 input channel, received "
                f"{x.shape[1]}."
            )

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.fc(x)

        return x


# ================================================================
# PREPROCESSING
#
# MUST MATCH TRAINING:
#
# Grayscale
# Resize 224x224
# ToTensor
# Normalize 0.485 / 0.229
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

    # ------------------------------------------------------------
    # DOWNLOAD FROM HUGGING FACE
    # ------------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
    )

    # ------------------------------------------------------------
    # CREATE EXACT TRAINING ARCHITECTURE
    # ------------------------------------------------------------

    model = ResNet3D18(
        num_classes=NUM_CLASSES
    )

    # ------------------------------------------------------------
    # LOAD CHECKPOINT
    # ------------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
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
    # REMOVE COMMON PREFIXES
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
    # ARCHITECTURE CHECK
    # ------------------------------------------------------------

    model_keys = set(
        model.state_dict().keys()
    )

    checkpoint_keys = set(
        cleaned_state_dict.keys()
    )

    missing = (
        model_keys
        - checkpoint_keys
    )

    unexpected = (
        checkpoint_keys
        - model_keys
    )

    if missing:

        raise RuntimeError(
            "Pneumonia checkpoint is missing "
            f"{len(missing)} model parameters.\n"
            f"Examples: {sorted(missing)[:10]}"
        )

    if unexpected:

        raise RuntimeError(
            "Pneumonia checkpoint contains "
            f"{len(unexpected)} unexpected parameters.\n"
            f"Examples: {sorted(unexpected)[:10]}"
        )

    # ------------------------------------------------------------
    # LOAD WEIGHTS
    # ------------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    # ------------------------------------------------------------
    # DEVICE + EVAL
    # ------------------------------------------------------------

    model = model.to(
        DEVICE
    )

    model.eval()

    _model = model

    return _model


# ================================================================
# CREATE PSEUDO-3D INPUT
#
# THIS IS THE IMPORTANT FIX.
#
# Input after transform:
#
#   [1,224,224]
#
# We explicitly create:
#
#   [1,1,224,224]
#
# then:
#
#   [1,1,16,224,224]
#
# There is NO ambiguity about which dimension is depth.
# ================================================================

def prepare_input(image):

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
    # 2D PREPROCESSING
    # ------------------------------------------------------------

    tensor = transform(
        image
    )

    # Expected:
    #
    # [1,224,224]

    if tensor.shape != (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):

        raise RuntimeError(
            "Unexpected 2D preprocessing "
            f"shape: {tuple(tensor.shape)}. "
            "Expected [1,224,224]."
        )

    # ------------------------------------------------------------
    # ADD BATCH DIMENSION
    #
    # [1,224,224]
    # ->
    # [1,1,224,224]
    #
    # This is [B,C,H,W].
    # ------------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    # ------------------------------------------------------------
    # ADD DEPTH DIMENSION
    #
    # [1,1,224,224]
    # ->
    # [1,1,1,224,224]
    #
    # This is [B,C,D,H,W].
    # ------------------------------------------------------------

    tensor = tensor.unsqueeze(
        2
    )

    # ------------------------------------------------------------
    # REPLICATE ALONG DEPTH
    #
    # [1,1,1,224,224]
    # ->
    # [1,1,16,224,224]
    # ------------------------------------------------------------

    tensor = tensor.repeat(
        1,
        1,
        DEPTH,
        1,
        1,
    )

    # ------------------------------------------------------------
    # FINAL VALIDATION
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
            "Pneumonia input construction "
            "failed.\n"
            f"Expected: {expected}\n"
            f"Received: {tuple(tensor.shape)}"
        )

    if tensor.ndim != 5:

        raise RuntimeError(
            "Pneumonia input MUST be 5D. "
            f"Received {tensor.ndim}D."
        )

    return tensor.float()


# ================================================================
# PREDICTION
# ================================================================

@torch.no_grad()
def predict(image):

    model = load_model()

    # ------------------------------------------------------------
    # PREPARE EXACT 5D INPUT
    # ------------------------------------------------------------

    tensor = prepare_input(
        image
    )

    # ------------------------------------------------------------
    # MOVE TO DEVICE
    # ------------------------------------------------------------

    tensor = tensor.to(
        DEVICE,
        non_blocking=True,
    )

    # ------------------------------------------------------------
    # FINAL CHECK IMMEDIATELY BEFORE MODEL
    # ------------------------------------------------------------

    if tensor.ndim != 5:

        raise RuntimeError(
            "FATAL: Tensor entering "
            "ResNet3D18 is not 5D.\n"
            f"Shape: {tuple(tensor.shape)}"
        )

    if tuple(tensor.shape) != (
        1,
        1,
        16,
        224,
        224,
    ):

        raise RuntimeError(
            "FATAL: Incorrect tensor entering "
            "Pneumonia model.\n"
            "Expected: [1,1,16,224,224]\n"
            f"Received: {tuple(tensor.shape)}"
        )

    # ------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------

    logits = model(
        tensor
    )

    # ------------------------------------------------------------
    # SOFTMAX
    # ------------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    # ------------------------------------------------------------
    # PROBABILITIES
    # ------------------------------------------------------------

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
    # RESULT
    # ------------------------------------------------------------

    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probability_dict,
    }
