import json
import re

import streamlit as st
import torch
import torch.nn as nn
import timm

from torchvision import transforms
from huggingface_hub import hf_hub_download


# ============================================================
# MEDUSA / MAMMOSENSE V2
# ============================================================

HF_REPO = "Makky07/MammoSense-breast-ultrasound"

MODEL_FILE = "mammosense_v2.pt"
CONFIG_FILE = "mammosense_v2_config.json"

IMAGE_SIZE = 224

DEFAULT_CLASSES = [
    "Normal",
    "Benign",
    "Malignant",
]


# ============================================================
# PYTORCH 2.6 SAFE LOADER
# ============================================================

def torch_load_safe(path):

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location="cpu",
        )


# ============================================================
# LOAD CONFIG
# ============================================================

@st.cache_data
def load_config():

    config_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=CONFIG_FILE,
        repo_type="model",
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# DOWNLOAD MODEL
# ============================================================

@st.cache_data
def download_checkpoint():

    return hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILE,
        repo_type="model",
    )


# ============================================================
# MAMMOSENSE V2
# ============================================================

class MammoSenseV2(nn.Module):

    def __init__(
        self,
        head_dimensions,
    ):

        super().__init__()

        # ----------------------------------------------------
        # EXACT BACKBONE
        # ----------------------------------------------------

        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,
        )

        feature_dim = (
            self.backbone.num_features
        )

        # ----------------------------------------------------
        # RECREATE:
        #
        # head.0
        # head.3
        # head.6
        #
        # Therefore:
        #
        # Linear
        # ReLU
        # Dropout
        # Linear
        # ReLU
        # Dropout
        # Linear
        # ----------------------------------------------------

        layers = []

        in_features = feature_dim

        for i, out_features in enumerate(
            head_dimensions
        ):

            layers.append(
                nn.Linear(
                    in_features,
                    out_features,
                )
            )

            if i < len(head_dimensions) - 1:

                layers.append(
                    nn.ReLU()
                )

                layers.append(
                    nn.Dropout(
                        p=0.0
                    )
                )

            in_features = out_features

        self.head = nn.Sequential(
            *layers
        )


    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# FIND HEAD DIMENSIONS
# ============================================================

def infer_head_dimensions(
    state_dict
):

    """
    Finds:

        head.0.weight
        head.3.weight
        head.6.weight

    regardless of the gaps between layer numbers.
    """

    head_layers = []

    pattern = re.compile(
        r"^head\.(\d+)\.weight$"
    )

    for key, tensor in state_dict.items():

        match = pattern.match(key)

        if match is None:
            continue

        layer_index = int(
            match.group(1)
        )

        if not torch.is_tensor(tensor):
            continue

        if tensor.ndim != 2:
            continue

        out_features = tensor.shape[0]
        in_features = tensor.shape[1]

        head_layers.append(
            (
                layer_index,
                in_features,
                out_features,
            )
        )

    # Sort by actual layer number

    head_layers.sort(
        key=lambda x: x[0]
    )

    if not head_layers:

        raise RuntimeError(
            "No classifier layers were found "
            "inside the MammoSense checkpoint."
        )

    # --------------------------------------------------------
    # Validate connections
    # --------------------------------------------------------

    dimensions = []

    previous_output = None

    for (
        layer_index,
        in_features,
        out_features,
    ) in head_layers:

        if (
            previous_output is not None
            and in_features != previous_output
        ):

            raise RuntimeError(
                "MammoSense classifier dimensions "
                "are inconsistent.\n\n"

                f"Layer: head.{layer_index}\n"
                f"Expected input: {previous_output}\n"
                f"Actual input: {in_features}"
            )

        dimensions.append(
            out_features
        )

        previous_output = out_features

    return dimensions


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = load_config()

    classes = config.get(
        "classes",
        DEFAULT_CLASSES,
    )

    image_size = int(
        config.get(
            "image_size",
            IMAGE_SIZE,
        )
    )

    architecture = config.get(
        "architecture",
        "vit_small_patch16_224",
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    model_path = (
        download_checkpoint()
    )

    checkpoint = torch_load_safe(
        model_path
    )

    # --------------------------------------------------------
    # STATE DICTIONARY
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    elif isinstance(
        checkpoint,
        dict,
    ):

        state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported MammoSense checkpoint."
        )

    # --------------------------------------------------------
    # CLEAN POSSIBLE PREFIX
    # --------------------------------------------------------

    cleaned_state = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        cleaned_state[
            new_key
        ] = value

    state_dict = cleaned_state

    # --------------------------------------------------------
    # VERIFY BACKBONE
    # --------------------------------------------------------

    backbone_keys = [
        key
        for key in state_dict
        if key.startswith(
            "backbone."
        )
    ]

    if not backbone_keys:

        raise RuntimeError(
            "MammoSense checkpoint does not "
            "contain backbone.* weights."
        )

    # --------------------------------------------------------
    # FIND ACTUAL HEAD
    # --------------------------------------------------------

    head_dimensions = (
        infer_head_dimensions(
            state_dict
        )
    )

    # --------------------------------------------------------
    # VERIFY OUTPUT
    # --------------------------------------------------------

    if (
        head_dimensions[-1]
        != len(classes)
    ):

        raise RuntimeError(
            "The classifier output does not "
            "match the configured classes.\n\n"

            f"Classifier output: "
            f"{head_dimensions[-1]}\n"

            f"Number of classes: "
            f"{len(classes)}"
        )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = MammoSenseV2(
        head_dimensions
    )

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "MammoSense V2 checkpoint "
            "still does not match the "
            "reconstructed model.\n\n"

            f"Architecture: "
            f"{architecture}\n\n"

            f"Head dimensions: "
            f"{head_dimensions}\n\n"

            f"Original error:\n"
            f"{error}"
        )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    model = model.to(
        device
    )

    model.eval()

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    transform = transforms.Compose(
        [

            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],

                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    return {

        "model": model,

        "transform": transform,

        "classes": list(
            classes
        ),

        "device": device,

        "architecture": architecture,

        "image_size": image_size,

        "model_file": MODEL_FILE,

        "head_dimensions":
            head_dimensions,

        "test_accuracy":
            config.get(
                "test_accuracy"
            ),

        "test_macro_f1":
            config.get(
                "test_macro_f1"
            ),

        "malignant_sensitivity":
            config.get(
                "malignant_sensitivity"
            ),

        "malignant_specificity":
            config.get(
                "malignant_specificity"
            ),
    }


# ============================================================
# PREDICTION
# ============================================================

@torch.inference_mode()
def predict(image):

    package = load_model()

    model = package[
        "model"
    ]

    transform = package[
        "transform"
    ]

    classes = package[
        "classes"
    ]

    device = package[
        "device"
    ]

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        device
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    predicted_index = int(
        torch.argmax(
            probabilities
        ).item()
    )

    prediction = classes[
        predicted_index
    ]

    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )

    # --------------------------------------------------------
    # ALL PROBABILITIES
    # --------------------------------------------------------

    probability_dict = {}

    for i, class_name in enumerate(
        classes
    ):

        probability_dict[
            class_name
        ] = float(
            probabilities[
                i
            ].item()
        )

    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probability_dict,
    }
