import json
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
# CHECKPOINT LOADER
# ============================================================

def torch_load_safe(path):

    """
    PyTorch 2.6 compatible checkpoint loader.

    The checkpoint is from the user's own trusted
    Hugging Face repository, so weights_only=False
    is intentional.
    """

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        # Older PyTorch versions

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

        config = json.load(f)

    return config


# ============================================================
# DOWNLOAD CHECKPOINT
# ============================================================

@st.cache_data
def download_checkpoint():

    return hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILE,
        repo_type="model",
    )


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class MammoSenseV2(nn.Module):

    def __init__(
        self,
        head_dimensions,
        num_classes=3,
    ):

        super().__init__()

        # ----------------------------------------------------
        # ViT BACKBONE
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
        # BUILD HEAD
        #
        # Example checkpoint:
        #
        # head.0
        # head.3
        # head.6
        #
        # Which corresponds to:
        #
        # Linear
        # activation
        # dropout
        # Linear
        # activation
        # dropout
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

            # Last layer = classifier

            if (
                i
                <
                len(head_dimensions) - 1
            ):

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

        features = self.backbone(
            x
        )

        return self.head(
            features
        )


# ============================================================
# INFER HEAD DIMENSIONS
# ============================================================

def infer_head_dimensions(
    state_dict,
):

    dimensions = []

    index = 0

    while True:

        weight_key = (
            f"head.{index}.weight"
        )

        bias_key = (
            f"head.{index}.bias"
        )

        if weight_key not in state_dict:

            break

        weight = state_dict[
            weight_key
        ]

        bias = state_dict.get(
            bias_key
        )

        if weight.ndim != 2:

            index += 1

            continue

        out_features = (
            weight.shape[0]
        )

        in_features = (
            weight.shape[1]
        )

        dimensions.append(
            (
                in_features,
                out_features,
            )
        )

        index += 1

    if not dimensions:

        raise RuntimeError(
            "Could not determine the "
            "MammoSense classifier head."
        )

    return dimensions


# ============================================================
# CREATE MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_model():

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = load_config()

    architecture = config.get(
        "architecture",
        "vit_small_patch16_224",
    )

    classes = config.get(
        "classes",
        DEFAULT_CLASSES,
    )

    image_size = config.get(
        "image_size",
        IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = (
        download_checkpoint()
    )

    checkpoint = torch_load_safe(
        checkpoint_path
    )

    # --------------------------------------------------------
    # EXTRACT STATE DICT
    # --------------------------------------------------------

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(
            checkpoint,
            dict,
        )
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

        # Raw state dictionary

        state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported MammoSense "
            "checkpoint format."
        )

    # --------------------------------------------------------
    # CONFIRM CUSTOM WRAPPER
    # --------------------------------------------------------

    backbone_keys = [
        k
        for k in state_dict
        if k.startswith(
            "backbone."
        )
    ]

    head_keys = [
        k
        for k in state_dict
        if k.startswith(
            "head."
        )
    ]

    if not backbone_keys:

        raise RuntimeError(
            "This checkpoint does not contain "
            "the expected 'backbone.*' weights."
        )

    if not head_keys:

        raise RuntimeError(
            "This checkpoint does not contain "
            "the expected 'head.*' weights."
        )

    # --------------------------------------------------------
    # INFER HEAD
    # --------------------------------------------------------

    dimensions = (
        infer_head_dimensions(
            state_dict
        )
    )

    head_dimensions = [
        out_features
        for (
            in_features,
            out_features
        )
        in dimensions
    ]

    # --------------------------------------------------------
    # VERIFY FINAL OUTPUT
    # --------------------------------------------------------

    if (
        head_dimensions[-1]
        != len(classes)
    ):

        raise RuntimeError(
            "Classifier output does not "
            "match the configuration.\n\n"

            f"Head output: "
            f"{head_dimensions[-1]}\n"

            f"Classes: "
            f"{len(classes)}"
        )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = MammoSenseV2(
        head_dimensions=head_dimensions,
        num_classes=len(classes),
    )

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as e:

        raise RuntimeError(
            "MammoSense V2 checkpoint "
            "could not be loaded.\n\n"

            f"Architecture: "
            f"{architecture}\n\n"

            f"Head dimensions: "
            f"{head_dimensions}\n\n"

            f"Error:\n{e}"
        )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    model = model.to(
        device
    )

    model.eval()

    # --------------------------------------------------------
    # TRANSFORM
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
        "classes": list(classes),
        "device": device,
        "architecture": architecture,
        "image_size": image_size,
        "model_file": MODEL_FILE,
        "head_dimensions": head_dimensions,
        "test_accuracy": config.get(
            "test_accuracy"
        ),
        "test_macro_f1": config.get(
            "test_macro_f1"
        ),
        "malignant_sensitivity": config.get(
            "malignant_sensitivity"
        ),
        "malignant_specificity": config.get(
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
    # INFERENCE
    # --------------------------------------------------------

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    index = int(
        torch.argmax(
            probabilities
        ).item()
    )

    prediction = classes[
        index
    ]

    confidence = float(
        probabilities[
            index
        ].item()
    )

    probability_dict = {}

    for i, class_name in enumerate(
        classes
    ):

        probability_dict[
            class_name
        ] = float(
            probabilities[i].item()
        )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_dict,
    }
