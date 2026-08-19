import json
import re

import streamlit as st
import torch
import torch.nn as nn
import timm

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download


# ============================================================
# MEDUSA / MAMMOSENSE V2
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
# CONFIG
# ============================================================

@st.cache_data(show_spinner=False)
def load_config():

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=CONFIG_FILE,
        repo_type="model",
    )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

@st.cache_data(show_spinner=False)
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

    def __init__(self, head_dimensions):

        super().__init__()

        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,
        )

        feature_dim = self.backbone.num_features

        layers = []

        in_features = feature_dim

        for i, out_features in enumerate(head_dimensions):

            layers.append(
                nn.Linear(
                    in_features,
                    out_features,
                )
            )

            if i < len(head_dimensions) - 1:

                layers.append(nn.ReLU())

                layers.append(
                    nn.Dropout(p=0.0)
                )

            in_features = out_features

        self.head = nn.Sequential(*layers)

    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# EXTRACT CHECKPOINT
# ============================================================

def extract_state_dict(checkpoint):

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise RuntimeError(
        "Unsupported MammoSense checkpoint format."
    )


# ============================================================
# CLEAN PREFIX
# ============================================================

def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


# ============================================================
# FIND HEAD DIMENSIONS
# ============================================================

def infer_head_dimensions(state_dict):

    layers = []

    pattern = re.compile(
        r"^head\.(\d+)\.weight$"
    )

    for key, tensor in state_dict.items():

        match = pattern.match(key)

        if match is None:
            continue

        if not torch.is_tensor(tensor):
            continue

        if tensor.ndim != 2:
            continue

        layer_number = int(match.group(1))

        layers.append(
            (
                layer_number,
                tensor.shape[1],
                tensor.shape[0],
            )
        )

    layers.sort(
        key=lambda x: x[0]
    )

    if not layers:

        raise RuntimeError(
            "No classifier layers found "
            "in MammoSense checkpoint."
        )

    dimensions = []

    previous_output = None

    for (
        layer_number,
        input_features,
        output_features,
    ) in layers:

        if (
            previous_output is not None
            and input_features != previous_output
        ):

            raise RuntimeError(
                "MammoSense classifier dimensions "
                "are inconsistent.\n\n"
                f"Layer: head.{layer_number}\n"
                f"Expected input: {previous_output}\n"
                f"Actual input: {input_features}"
            )

        dimensions.append(output_features)

        previous_output = output_features

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

    classes = list(
        config.get(
            "classes",
            DEFAULT_CLASSES,
        )
    )

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
    # DOWNLOAD
    # --------------------------------------------------------

    model_path = download_checkpoint()

    checkpoint = torch_load_safe(
        model_path
    )

    # --------------------------------------------------------
    # STATE DICT
    # --------------------------------------------------------

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    # --------------------------------------------------------
    # CHECK STRUCTURE
    # --------------------------------------------------------

    backbone_keys = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }

    head_keys = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("head.")
    }

    if not backbone_keys:

        raise RuntimeError(
            "No backbone.* weights found."
        )

    if not head_keys:

        raise RuntimeError(
            "No head.* weights found."
        )

    # --------------------------------------------------------
    # HEAD DIMENSIONS
    # --------------------------------------------------------

    head_dimensions = infer_head_dimensions(
        state_dict
    )

    if head_dimensions[-1] != len(classes):

        raise RuntimeError(
            "Classifier output does not match "
            "the number of classes.\n\n"
            f"Output: {head_dimensions[-1]}\n"
            f"Classes: {len(classes)}"
        )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = MammoSenseV2(
        head_dimensions
    )

    # ========================================================
    # IMPORTANT:
    #
    # CHECKPOINT:
    #
    # backbone.cls_token
    # backbone.pos_embed
    # backbone.blocks.0...
    #
    # ViT expects:
    #
    # cls_token
    # pos_embed
    # blocks.0...
    #
    # Therefore remove "backbone.".
    # ========================================================

    backbone_state = {}

    for key, value in backbone_keys.items():

        new_key = key[
            len("backbone.") :
        ]

        backbone_state[
            new_key
        ] = value

    # ========================================================
    # HEAD:
    #
    # checkpoint:
    #
    # head.0.weight
    # head.3.weight
    # head.6.weight
    #
    # Sequential head expects:
    #
    # 0.weight
    # 3.weight
    # 6.weight
    # ========================================================

    head_state = {}

    for key, value in head_keys.items():

        new_key = key[
            len("head.") :
        ]

        head_state[
            new_key
        ] = value

    # --------------------------------------------------------
    # LOAD BACKBONE SEPARATELY
    # --------------------------------------------------------

    try:

        model.backbone.load_state_dict(
            backbone_state,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "MammoSense ViT backbone failed "
            "to load.\n\n"
            f"Architecture: {architecture}\n\n"
            f"{error}"
        )

    # --------------------------------------------------------
    # LOAD HEAD SEPARATELY
    # --------------------------------------------------------

    try:

        model.head.load_state_dict(
            head_state,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "MammoSense classification head "
            "failed to load.\n\n"
            f"Head dimensions: {head_dimensions}\n\n"
            f"{error}"
        )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    model = model.to(device)

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

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

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
    # IMAGE
    # --------------------------------------------------------

    if not isinstance(image, Image.Image):

        image = Image.open(image)

    image = image.convert("RGB")

    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    logits = model(tensor)

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

    for i, class_name in enumerate(classes):

        probability_dict[
            class_name
        ] = float(
            probabilities[i].item()
        )

    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probability_dict,
    }
