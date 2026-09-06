# ============================================================
# MEDUSA AI
# DIABETIC RETINOPATHY MODEL
#
# Model:
#   EfficientNet-B3
#
# Classes:
#   0 = No DR
#   1 = Mild + Moderate NPDR
#   2 = Severe NPDR + PDR
#
# Output:
# {
#     "prediction": "...",
#     "confidence": 0.95,
#     "probabilities": {
#         "...": 0.95,
#         "...": 0.04,
#         "...": 0.01
#     }
# }
# ============================================================

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "diabetic_retinopathy"
)

MODEL_FILE = (
    MODEL_DIR
    / "MEDUSA_DDR_EfficientNetB3_3CLASS_5000_best.pt"
)

JSON_FILE = (
    MODEL_DIR
    / "MEDUSA_DDR_EfficientNetB3_3CLASS_5000.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "No DR",
    "Mild + Moderate NPDR",
    "Severe NPDR + PDR",
]


# ============================================================
# IMAGE TRANSFORM
#
# MUST MATCH TRAINING
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize(
        (300, 300)
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
])


# ============================================================
# MODEL CACHE
# ============================================================

_model = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global _model

    # --------------------------------------------------------
    # Already loaded
    # --------------------------------------------------------

    if _model is not None:

        return _model

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Diabetic retinopathy model "
            f"not found:\n{MODEL_FILE}"
        )

    # --------------------------------------------------------
    # Create EfficientNet-B3
    # --------------------------------------------------------

    model = timm.create_model(
        "tf_efficientnet_b3",
        pretrained=False,
        num_classes=3,
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
    )

    # --------------------------------------------------------
    # Handle different checkpoint formats
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        if "state_dict" in checkpoint:

            state_dict = (
                checkpoint["state_dict"]
            )

        elif "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint["model_state_dict"]
            )

        elif "model" in checkpoint:

            state_dict = (
                checkpoint["model"]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove common prefixes
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
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )

    model.eval()

    _model = model

    return _model


# ============================================================
# PREDICT
# ============================================================

def predict(image):

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if image is None:

        raise ValueError(
            "No retinal fundus image was provided."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    if not isinstance(
        image,
        Image.Image,
    ):

        image = Image.open(
            image
        )

    image = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    tensor = TRANSFORM(
        image
    )

    tensor = tensor.unsqueeze(
        0
    ).to(
        DEVICE
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        if DEVICE.type == "cuda":

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(
                    tensor
                )

        else:

            outputs = model(
                tensor
            )

        probabilities = F.softmax(
            outputs,
            dim=1,
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_class = (
        probabilities.argmax(
            dim=1
        ).item()
    )

    confidence = (
        probabilities[
            0,
            predicted_class
        ].item()
    )

    # --------------------------------------------------------
    # Probability dictionary
    # --------------------------------------------------------

    probability_dict = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probability_dict[
            class_name
        ] = round(
            probabilities[
                0,
                index
            ].item(),
            6,
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "prediction":
            CLASS_NAMES[
                predicted_class
            ],

        "confidence":
            float(
                confidence
            ),

        "probabilities":
            probability_dict,
    }


# ============================================================
# OPTIONAL MODEL INFORMATION
# ============================================================

def model_info():

    return {
        "model":
            "EfficientNet-B3",

        "model_file":
            str(
                MODEL_FILE
            ),

        "metadata_file":
            str(
                JSON_FILE
            ),

        "num_classes":
            3,

        "classes":
            CLASS_NAMES,

        "input_size":
            [300, 300],

        "device":
            str(
                DEVICE
            ),
    }
