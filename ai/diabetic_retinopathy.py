# ============================================================
# MEDUSA AI
# DIABETIC RETINOPATHY
#
# Model:
#   EfficientNet-B3
#
# Hugging Face:
#   Makky07/Retinopathy
#
# Classes:
#   0 = No DR
#   1 = Mild + Moderate NPDR
#   2 = Severe NPDR + PDR
# ============================================================

from pathlib import Path
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm


# ============================================================
# CONFIGURATION
# ============================================================

HF_MODEL_URL = (
    "https://huggingface.co/"
    "Makky07/Retinopathy/resolve/main/"
    "MEDUSA_DDR_EfficientNetB3_3CLASS_5000_best.pt"
)

HF_JSON_URL = (
    "https://huggingface.co/"
    "Makky07/Retinopathy/resolve/main/"
    "MEDUSA_DDR_EfficientNetB3_3CLASS_5000%20%281%29.json"
)

# Streamlit server cache location
CACHE_DIR = (
    Path.home()
    / ".cache"
    / "medusa"
    / "diabetic_retinopathy"
)

MODEL_FILE = (
    CACHE_DIR
    / "MEDUSA_DDR_EfficientNetB3_3CLASS_5000_best.pt"
)

JSON_FILE = (
    CACHE_DIR
    / "MEDUSA_DDR_EfficientNetB3_3CLASS_5000.json"
)

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
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# MODEL CACHE
# ============================================================

_model = None


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def _download_file(url, destination):

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Already downloaded
    if destination.exists():

        # Make sure it isn't an empty/partial file
        if destination.stat().st_size > 0:
            return destination

        destination.unlink()

    try:

        import requests

    except ImportError as e:

        raise ImportError(
            "The requests package is required "
            "to download the diabetic retinopathy model."
        ) from e

    print(
        "Downloading Medusa DR model "
        "from Hugging Face..."
    )

    response = requests.get(
        url,
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    total_size = int(
        response.headers.get(
            "content-length",
            0,
        )
    )

    downloaded = 0

    with open(
        destination,
        "wb",
    ) as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if not chunk:
                continue

            file.write(chunk)

            downloaded += len(chunk)

    # Verify something was actually downloaded
    if not destination.exists():

        raise RuntimeError(
            "Model download failed: "
            "file was not created."
        )

    if destination.stat().st_size == 0:

        destination.unlink()

        raise RuntimeError(
            "Model download failed: "
            "downloaded file is empty."
        )

    # Optional size sanity check
    if total_size > 0:

        actual_size = destination.stat().st_size

        if actual_size != total_size:

            destination.unlink()

            raise RuntimeError(
                "Incomplete model download.\n"
                f"Expected: {total_size:,} bytes\n"
                f"Received: {actual_size:,} bytes"
            )

    print(
        "Medusa DR model downloaded successfully."
    )

    return destination


# ============================================================
# ENSURE MODEL EXISTS
# ============================================================

def _ensure_model():

    if MODEL_FILE.exists():

        if MODEL_FILE.stat().st_size > 0:
            return MODEL_FILE

    return _download_file(
        HF_MODEL_URL,
        MODEL_FILE,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global _model

    if _model is not None:
        return _model

    # --------------------------------------------------------
    # Download model if necessary
    # --------------------------------------------------------

    model_path = _ensure_model()

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
        model_path,
        map_location="cpu",
    )

    # --------------------------------------------------------
    # Handle different checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "model" in checkpoint:

            state_dict = checkpoint["model"]

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

        if new_key.startswith("module."):
            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith("model."):
            new_key = new_key[
                len("model.") :
            ]

        cleaned_state_dict[new_key] = value

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

    model = model.to(DEVICE)

    model.eval()

    _model = model

    return _model


# ============================================================
# PREDICT
# ============================================================

def predict(image):

    if image is None:

        raise ValueError(
            "No retinal fundus image was provided."
        )

    model = load_model()

    # --------------------------------------------------------
    # Convert image
    # --------------------------------------------------------

    if not isinstance(
        image,
        Image.Image,
    ):

        image = Image.open(image)

    image = image.convert("RGB")

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    tensor = TRANSFORM(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        if DEVICE.type == "cuda":

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(tensor)

        else:

            outputs = model(tensor)

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = F.softmax(
        outputs,
        dim=1,
    )

    predicted_class = (
        probabilities
        .argmax(dim=1)
        .item()
    )

    confidence = (
        probabilities[
            0,
            predicted_class,
        ].item()
    )

    # --------------------------------------------------------
    # Probability dictionary
    # --------------------------------------------------------

    probability_dict = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probability_dict[class_name] = round(
            probabilities[
                0,
                index,
            ].item(),
            6,
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "prediction":
            CLASS_NAMES[predicted_class],

        "confidence":
            float(confidence),

        "probabilities":
            probability_dict,

    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_info():

    return {

        "model":
            "EfficientNet-B3",

        "architecture":
            "tf_efficientnet_b3",

        "source":
            "Hugging Face",

        "repository":
            "Makky07/Retinopathy",

        "model_file":
            str(MODEL_FILE),

        "metadata_file":
            str(JSON_FILE),

        "num_classes":
            3,

        "classes":
            CLASS_NAMES,

        "input_size":
            [300, 300],

        "device":
            str(DEVICE),

    }
