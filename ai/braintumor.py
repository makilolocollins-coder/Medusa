# ============================================================
# MEDUSA AI
# MAMMOSENSE BRAIN MRI V3.1
#
# 2D BRAIN MRI CLASSIFICATION
#
# Classes:
#   0 = glioma
#   1 = meningioma
#   2 = pituitary
#   3 = notumor
#
# Architecture:
#   ResNet-50
#
# Input:
#   Single 2D MRI image
#   RGB
#   224 x 224
#
# Hugging Face:
#   Makky07/BRAIN_MRI
#
# Checkpoint:
#   mammosense_brain_classifier_best.pt
# ============================================================

import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = "Makky07/BRAIN_MRI"

MODEL_FILENAME = (
    "mammosense_brain_classifier_best.pt"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor",
]

CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "notumor": 3,
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# GLOBAL MODEL
# ============================================================

_model = None


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
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
# BUILD MODEL
# ============================================================

def build_model():

    # --------------------------------------------------------
    # ResNet-50
    #
    # The checkpoint contains:
    #
    # layer1 -> 3 blocks
    # layer2 -> 4 blocks
    # layer3 -> 6 blocks
    # layer4 -> 3 blocks
    #
    # This matches ResNet-50.
    # --------------------------------------------------------

    model = models.resnet50(
        weights=None
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Checkpoint:
    #
    # fc.1.weight -> (4, 2048)
    # fc.1.bias   -> (4,)
    #
    # Therefore the original classifier was a Sequential
    # module whose final Linear layer was fc.1.
    # --------------------------------------------------------

    model.fc = nn.Sequential(
        nn.Dropout(
            p=0.5
        ),
        nn.Linear(
            2048,
            4
        ),
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global _model

    if _model is not None:

        return _model


    # --------------------------------------------------------
    # DOWNLOAD CHECKPOINT
    # --------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
    )


    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )


    # --------------------------------------------------------
    # CHECK CHECKPOINT
    # --------------------------------------------------------

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Invalid Brain MRI checkpoint. "
            "Expected a dictionary."
        )


    # --------------------------------------------------------
    # EXTRACT STATE DICTIONARY
    # --------------------------------------------------------

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:

        state_dict = checkpoint.get(
            "state_dict"
        )


    if state_dict is None:

        raise RuntimeError(
            "Brain MRI checkpoint does not "
            "contain model_state_dict."
        )


    # --------------------------------------------------------
    # REMOVE POSSIBLE PREFIXES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # BUILD ARCHITECTURE
    # --------------------------------------------------------

    model = build_model()


    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

    except RuntimeError as error:

        raise RuntimeError(
            "Brain MRI model architecture "
            "does not match the checkpoint.\n\n"
            f"{error}"
        )


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # EVALUATION MODE
    # --------------------------------------------------------

    model.eval()


    # --------------------------------------------------------
    # STORE GLOBAL MODEL
    # --------------------------------------------------------

    _model = model


    return _model


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image):

    # --------------------------------------------------------
    # ACCEPT PIL IMAGE
    # --------------------------------------------------------

    if isinstance(
        image,
        Image.Image,
    ):

        pil_image = image


    # --------------------------------------------------------
    # ACCEPT FILE PATH
    # --------------------------------------------------------

    elif isinstance(
        image,
        str,
    ):

        if not os.path.exists(
            image
        ):

            raise FileNotFoundError(
                f"Image not found: {image}"
            )

        pil_image = Image.open(
            image
        )


    # --------------------------------------------------------
    # ACCEPT BYTES
    # --------------------------------------------------------

    elif isinstance(
        image,
        bytes,
    ):

        import io

        pil_image = Image.open(
            io.BytesIO(
                image
            )
        )


    else:

        raise TypeError(
            "Brain MRI input must be "
            "a PIL Image, file path, "
            "or image bytes."
        )


    # --------------------------------------------------------
    # CONVERT TO RGB
    # --------------------------------------------------------

    pil_image = pil_image.convert(
        "RGB"
    )


    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    tensor = transform(
        pil_image
    )


    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    #
    # [3,224,224]
    #
    # becomes
    #
    # [1,3,224,224]
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )


    return tensor.to(
        DEVICE
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # PREPARE IMAGE
    # --------------------------------------------------------

    tensor = prepare_image(
        image
    )


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            tensor
        )


        # ----------------------------------------------------
        # SOFTMAX
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1,
        )


        # ----------------------------------------------------
        # BEST CLASS
        # ----------------------------------------------------

        confidence, predicted_idx = (
            torch.max(
                probabilities,
                dim=1,
            )
        )


    # --------------------------------------------------------
    # CONVERT VALUES
    # --------------------------------------------------------

    predicted_idx = int(
        predicted_idx.item()
    )

    confidence = float(
        confidence.item()
    )


    # --------------------------------------------------------
    # PREDICTION LABEL
    # --------------------------------------------------------

    prediction = CLASS_NAMES[
        predicted_idx
    ]


    # --------------------------------------------------------
    # PROBABILITY BREAKDOWN
    # --------------------------------------------------------

    probability_dict = {}

    for idx, class_name in enumerate(
        CLASS_NAMES
    ):

        probability_dict[
            class_name
        ] = float(
            probabilities[
                0,
                idx,
            ].item()
        )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "prediction": prediction,

        "confidence": confidence,

        "probabilities": probability_dict,

        "class_index": predicted_idx,

        "model": "MammoSense Brain V3.1",

        "architecture": "ResNet-50",

        "image_size": IMAGE_SIZE,
    }
