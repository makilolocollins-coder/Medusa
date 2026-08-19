# ============================================================
# MEDUSA AI
# MammoSense V2 Breast Ultrasound Model
# Streamlit + Hugging Face
# ============================================================

import json

import streamlit as st
import torch
import torch.nn as nn
import timm

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO = "Makky07/MammoSense-breast-ultrasound"

MODEL_FILE = "mammosense_v2.pt"
CONFIG_FILE = "mammosense_v2_config.json"

DEFAULT_IMAGE_SIZE = 224

DEFAULT_CLASSES = [
    "Normal",
    "Benign",
    "Malignant",
]


# ============================================================
# SAFE TORCH LOAD
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
# LOAD CONFIGURATION
# ============================================================

@st.cache_data(show_spinner=False)
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
    ) as file:

        config = json.load(file)

    return config


# ============================================================
# DOWNLOAD CHECKPOINT
# ============================================================

@st.cache_data(show_spinner=False)
def download_checkpoint():

    return hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILE,
        repo_type="model",
    )


# ============================================================
# MAMMOSENSE V2 ARCHITECTURE
# ============================================================

class MammoSenseV2(nn.Module):

    def __init__(
        self,
        head_dimensions,
    ):

        super().__init__()

        # ----------------------------------------------------
        # ViT-Small backbone
        # ----------------------------------------------------

        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,
        )

        feature_dim = self.backbone.num_features

        # ----------------------------------------------------
        # Custom classification head
        #
        # head.0
        # head.3
        # head.6
        # ----------------------------------------------------

        layers = []

        in_features = feature_dim

        for index, out_features in enumerate(
            head_dimensions
        ):

            layers.append(
                nn.Linear(
                    in_features,
                    out_features,
                )
            )

            # Hidden layers

            if index < len(head_dimensions) - 1:

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
# FIND CLASSIFIER DIMENSIONS
# ============================================================

def infer_head_dimensions(state_dict):

    head_layers = []

    for key, tensor in state_dict.items():

        if not key.startswith("head."):
            continue

        if not key.endswith(".weight"):
            continue

        if not torch.is_tensor(tensor):
            continue

        if tensor.ndim != 2:
            continue

        try:

            layer_number = int(
                key.split(".")[1]
            )

        except (IndexError, ValueError):

            continue

        out_features = tensor.shape[0]
        in_features = tensor.shape[1]

        head_layers.append(
            (
                layer_number,
                in_features,
                out_features,
            )
        )

    head_layers.sort(
        key=lambda x: x[0]
    )

    if not head_layers:

        raise RuntimeError(
            "No classification layers were found "
            "in the MammoSense checkpoint."
        )

    dimensions = []

    previous_output = None

    for (
        layer_number,
        in_features,
        out_features,
    ) in head_layers:

        if previous_output is not None:

            if in_features != previous_output:

                raise RuntimeError(
                    "MammoSense classifier dimensions "
                    "are inconsistent.\n\n"
                    f"Layer: head.{layer_number}\n"
                    f"Expected input: {previous_output}\n"
                    f"Actual input: {in_features}"
                )

        dimensions.append(
            out_features
        )

        previous_output = out_features

    return dimensions


# ============================================================
# EXTRACT STATE DICTIONARY
# ============================================================

def extract_state_dict(checkpoint):

    # --------------------------------------------------------
    # Standard training checkpoint
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        return checkpoint["model_state_dict"]

    # --------------------------------------------------------
    # Alternative checkpoint
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        return checkpoint["state_dict"]

    # --------------------------------------------------------
    # Raw state dictionary
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        tensor_values = [
            value
            for value in checkpoint.values()
            if torch.is_tensor(value)
        ]

        if tensor_values:

            return checkpoint

    raise RuntimeError(
        "MammoSense checkpoint format is not supported."
    )


# ============================================================
# REMOVE MODULE PREFIX
# ============================================================

def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):

            new_key = new_key[
                len("module.") :
            ]

        cleaned[new_key] = value

    return cleaned


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_model():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config()

    classes = config.get(
        "classes",
        DEFAULT_CLASSES,
    )

    classes = list(classes)

    image_size = int(
        config.get(
            "image_size",
            DEFAULT_IMAGE_SIZE,
        )
    )

    architecture = config.get(
        "architecture",
        "vit_small_patch16_224",
    )

    # --------------------------------------------------------
    # Download checkpoint
    # --------------------------------------------------------

    model_path = download_checkpoint()

    checkpoint = torch_load_safe(
        model_path
    )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    # --------------------------------------------------------
    # Check checkpoint structure
    # --------------------------------------------------------

    has_backbone = any(
        key.startswith("backbone.")
        for key in state_dict
    )

    has_head = any(
        key.startswith("head.")
        for key in state_dict
    )

    if not has_backbone:

        raise RuntimeError(
            "MammoSense checkpoint does not contain "
            "'backbone.*' weights."
        )

    if not has_head:

        raise RuntimeError(
            "MammoSense checkpoint does not contain "
            "'head.*' weights."
        )

    # --------------------------------------------------------
    # Infer classifier
    # --------------------------------------------------------

    head_dimensions = infer_head_dimensions(
        state_dict
    )

    # --------------------------------------------------------
    # Verify number of classes
    # --------------------------------------------------------

    if head_dimensions[-1] != len(classes):

        raise RuntimeError(
            "MammoSense classifier output does not "
            "match the configured classes.\n\n"
            f"Classifier output: "
            f"{head_dimensions[-1]}\n"
            f"Classes: "
            f"{len(classes)}"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MammoSenseV2(
        head_dimensions=head_dimensions
    )

    # ========================================================
    # IMPORTANT FIX
    #
    # The checkpoint contains:
    #
    # backbone.cls_token
    # backbone.pos_embed
    # backbone.blocks...
    #
    # But the ViT itself expects:
    #
    # cls_token
    # pos_embed
    # blocks...
    #
    # Therefore we strip "backbone." before loading
    # the backbone.
    # ========================================================

    backbone_state = {}

    head_state = {}

    for key, value in state_dict.items():

        if key.startswith("backbone."):

            new_key = key[
                len("backbone.") :
            ]

            backbone_state[
                new_key
            ] = value

        elif key.startswith("head."):

            new_key = key[
                len("head.") :
            ]

            head_state[
                new_key
            ] = value

    # --------------------------------------------------------
    # Load backbone
    # --------------------------------------------------------

    try:

        model.backbone.load_state_dict(
            backbone_state,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "MammoSense ViT backbone could not "
            "be loaded.\n\n"
            f"Architecture: {architecture}\n\n"
            f"Original error:\n{error}"
        )

    # --------------------------------------------------------
    # Load classification head
    # --------------------------------------------------------

    try:

        model.head.load_state_dict(
            head_state,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "MammoSense classification head could "
            "not be loaded.\n\n"
            f"Head dimensions: {head_dimensions}\n\n"
            f"Original error:\n{error}"
        )

    # --------------------------------------------------------
    # Move model to device
    # --------------------------------------------------------

    model = model.to(
        device
    )

    model.eval()

    # ========================================================
    # IMAGE PREPROCESSING
    # ========================================================

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

    # ========================================================
    # RETURN MODEL PACKAGE
    # ========================================================

    return {

        "model": model,

        "transform": transform,

        "classes": classes,

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

    model = package["model"]

    transform = package["transform"]

    classes = package["classes"]

    device = package["device"]

    # --------------------------------------------------------
    # Ensure PIL image
    # --------------------------------------------------------

    if not isinstance(image, Image.Image):

        image = Image.open(
            image
        )

    image = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

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
    # Inference
    # --------------------------------------------------------

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

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
    # Probability table
    # --------------------------------------------------------

    probability_dict = {}

    for index, class_name in enumerate(
        classes
    ):

        probability_dict[
            class_name
        ] = float(
            probabilities[
                index
            ].item()
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probability_dict,
    }


# ============================================================
# MODEL STATUS
# ============================================================

def get_model_info():

    package = load_model()

    return {

        "architecture":
            package["architecture"],

        "classes":
            package["classes"],

        "image_size":
            package["image_size"],

        "head_dimensions":
            package["head_dimensions"],

        "test_accuracy":
            package["test_accuracy"],

        "test_macro_f1":
            package["test_macro_f1"],

        "malignant_sensitivity":
            package[
                "malignant_sensitivity"
            ],

        "malignant_specificity":
            package[
                "malignant_specificity"
            ],
    }
