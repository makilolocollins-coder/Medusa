# ============================================================
# MEDUSA AI
# MammoSense V2 Model Engine
#
# This file contains ONLY:
#   - Hugging Face model download
#   - MammoSense architecture
#   - checkpoint loading
#   - image preprocessing
#   - prediction
#
# UI is intentionally kept separate.
# ============================================================

import json

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

DEFAULT_ARCHITECTURE = "vit_small_patch16_224"

DEFAULT_CLASSES = [
    "Normal",
    "Benign",
    "Malignant",
]

DEFAULT_IMAGE_SIZE = 224


# ============================================================
# PYTORCH CHECKPOINT LOADER
# ============================================================

def _load_checkpoint(path):

    """
    PyTorch 2.6-compatible loader.

    The checkpoint is from the user's own trusted
    Hugging Face repository.
    """

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
# DOWNLOAD CONFIG
# ============================================================

def _download_config():

    return hf_hub_download(
        repo_id=HF_REPO,
        filename=CONFIG_FILE,
        repo_type="model",
    )


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def _download_model():

    return hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILE,
        repo_type="model",
    )


# ============================================================
# READ CONFIG
# ============================================================

def _read_config():

    try:

        path = _download_config()

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return {
            "architecture": DEFAULT_ARCHITECTURE,
            "classes": DEFAULT_CLASSES,
            "image_size": DEFAULT_IMAGE_SIZE,
        }


# ============================================================
# MAMMOSENSE V2
# ============================================================

class MammoSenseV2(nn.Module):

    """
    Exact high-level structure of the checkpoint:

        backbone.*
        head.*

    The backbone is:

        timm ViT-Small Patch16 224

    The classifier is reconstructed from
    the checkpoint's head.* tensors.
    """

    def __init__(self, head_dimensions):

        super().__init__()

        # ----------------------------------------------------
        # BACKBONE
        # ----------------------------------------------------

        self.backbone = timm.create_model(
            DEFAULT_ARCHITECTURE,
            pretrained=False,
            num_classes=0,
        )

        feature_dim = self.backbone.num_features

        # ----------------------------------------------------
        # CLASSIFICATION HEAD
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

            # Every layer except the final
            # classification layer is followed by:
            #
            # ReLU
            # Dropout
            #

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

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# EXTRACT STATE DICTIONARY
# ============================================================

def _extract_state_dict(checkpoint):

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "MammoSense checkpoint is not a dictionary."
        )

    # Common checkpoint format

    if "model_state_dict" in checkpoint:

        return checkpoint["model_state_dict"]

    # Alternative format

    if "state_dict" in checkpoint:

        return checkpoint["state_dict"]

    # Raw state dictionary

    tensor_values = [
        value
        for value in checkpoint.values()
        if torch.is_tensor(value)
    ]

    if tensor_values:

        return checkpoint

    raise RuntimeError(
        "Could not find a model state dictionary "
        "inside mammosense_v2.pt."
    )


# ============================================================
# CLEAN STATE DICTIONARY
# ============================================================

def _clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        # Remove DataParallel prefix

        if new_key.startswith("module."):

            new_key = new_key[
                len("module.") :
            ]

        cleaned[new_key] = value

    return cleaned


# ============================================================
# FIND CLASSIFIER DIMENSIONS
# ============================================================

def _get_head_dimensions(state_dict):

    """
    Looks specifically for:

        head.0.weight
        head.3.weight
        head.6.weight

    The actual tensor shapes determine
    the classifier architecture.
    """

    head_weights = []

    for key, tensor in state_dict.items():

        if not key.startswith("head."):

            continue

        if not key.endswith(".weight"):

            continue

        if not torch.is_tensor(tensor):

            continue

        if tensor.ndim != 2:

            continue

        layer_number = int(
            key.split(".")[1]
        )

        head_weights.append(
            (
                layer_number,
                tensor.shape[1],
                tensor.shape[0],
            )
        )

    head_weights.sort(
        key=lambda item: item[0]
    )

    if not head_weights:

        raise RuntimeError(
            "No head.* Linear layers were found "
            "in the MammoSense checkpoint."
        )

    dimensions = []

    previous_output = None

    for (
        layer_number,
        input_features,
        output_features,
    ) in head_weights:

        if previous_output is not None:

            if input_features != previous_output:

                raise RuntimeError(
                    "MammoSense classifier structure "
                    "is inconsistent.\n\n"

                    f"head.{layer_number}: "
                    f"expected {previous_output} input features, "
                    f"but checkpoint contains "
                    f"{input_features}."
                )

        dimensions.append(
            output_features
        )

        previous_output = output_features

    return dimensions


# ============================================================
# VERIFY CHECKPOINT STRUCTURE
# ============================================================

def _verify_checkpoint(state_dict):

    backbone_found = any(
        key.startswith("backbone.")
        for key in state_dict
    )

    head_found = any(
        key.startswith("head.")
        for key in state_dict
    )

    if not backbone_found:

        raise RuntimeError(
            "MammoSense checkpoint does not contain "
            "'backbone.*' weights."
        )

    if not head_found:

        raise RuntimeError(
            "MammoSense checkpoint does not contain "
            "'head.*' weights."
        )


# ============================================================
# BUILD MODEL
# ============================================================

def _build_model(state_dict):

    head_dimensions = _get_head_dimensions(
        state_dict
    )

    model = MammoSenseV2(
        head_dimensions
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The checkpoint already has:
    #
    # backbone.cls_token
    # backbone.pos_embed
    # backbone.patch_embed...
    #
    # therefore we load it into the complete
    # MammoSenseV2 structure.
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "MammoSense checkpoint could not be "
            "loaded into the reconstructed MammoSenseV2 "
            "architecture.\n\n"
            f"Detected head dimensions: "
            f"{head_dimensions}\n\n"
            f"PyTorch error:\n{error}"
        )

    return model, head_dimensions


# ============================================================
# PREPROCESSING
# ============================================================

def _create_transform(image_size):

    return transforms.Compose(
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


# ============================================================
# MAMMOSENSE ENGINE
# ============================================================

class MammoSense:

    def __init__(self):

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ----------------------------------------------------
        # CONFIG
        # ----------------------------------------------------

        self.config = _read_config()

        self.classes = self.config.get(
            "classes",
            DEFAULT_CLASSES,
        )

        self.architecture = self.config.get(
            "architecture",
            DEFAULT_ARCHITECTURE,
        )

        self.image_size = int(
            self.config.get(
                "image_size",
                DEFAULT_IMAGE_SIZE,
            )
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        model_path = _download_model()

        # ----------------------------------------------------
        # LOAD CHECKPOINT
        # ----------------------------------------------------

        checkpoint = _load_checkpoint(
            model_path
        )

        state_dict = _extract_state_dict(
            checkpoint
        )

        state_dict = _clean_state_dict(
            state_dict
        )

        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------

        _verify_checkpoint(
            state_dict
        )

        # ----------------------------------------------------
        # BUILD
        # ----------------------------------------------------

        self.model, self.head_dimensions = (
            _build_model(
                state_dict
            )
        )

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        self.transform = _create_transform(
            self.image_size
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        self.test_accuracy = self.config.get(
            "test_accuracy"
        )

        self.test_macro_f1 = self.config.get(
            "test_macro_f1"
        )

        self.malignant_sensitivity = (
            self.config.get(
                "malignant_sensitivity"
            )
        )

        self.malignant_specificity = (
            self.config.get(
                "malignant_specificity"
            )
        )


# ============================================================
# PREDICTION
# ============================================================

    @torch.inference_mode()
    def predict(self, image):

        # ----------------------------------------------------
        # PIL IMAGE
        # ----------------------------------------------------

        if not isinstance(
            image,
            Image.Image,
        ):

            raise TypeError(
                "MammoSense expects a PIL Image."
            )

        image = image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            self.device
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        logits = self.model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        prediction = self.classes[
            predicted_index
        ]

        confidence = float(
            probabilities[
                predicted_index
            ].item()
        )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probability_dict = {}

        for index, class_name in enumerate(
            self.classes
        ):

            probability_dict[
                class_name
            ] = float(
                probabilities[
                    index
                ].item()
            )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {

            "prediction": prediction,

            "confidence": confidence,

            "probabilities": probability_dict,

            "architecture": self.architecture,

            "model": "MammoSense V2",

            "device": str(
                self.device
            ),
        }


# ============================================================
# SINGLETON HELPER
# ============================================================

_ENGINE = None


def get_mammosense():

    """
    Returns one MammoSense engine instance.

    The UI can simply use:

        engine = get_mammosense()
        result = engine.predict(image)
    """

    global _ENGINE

    if _ENGINE is None:

        _ENGINE = MammoSense()

    return _ENGINE
