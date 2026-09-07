# ============================================================
# MEDUSA AI
# DIABETIC RETINOPATHY
#
# Hierarchical 2-Stage EfficientNet-B3
#
# Stage 1:
#   No DR vs Any DR
#   Architecture: timm EfficientNet-B3
#
# Stage 2:
#   Mild + Moderate vs Severe + PDR
#   Architecture: torchvision EfficientNet-B3
#
# Hugging Face:
#   Makky07/Retinopathy
#
# Final classes:
#   0 = No DR
#   1 = Mild + Moderate NPDR
#   2 = Severe NPDR + PDR
# ============================================================

from pathlib import Path
import json

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3

import timm


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO = "Makky07/Retinopathy"

STAGE1_FILENAME = (
    "MEDUSA_DR_STAGE1_NoDR_vs_DR_best.pt"
)

STAGE2_FILENAME = (
    "MEDUSA_DR_STAGE2_V2_MildModerate_vs_"
    "SeverePDR_best.pt"
)

THRESHOLD_FILENAME = (
    "MEDUSA_DR_STAGE2_V2_threshold.json"
)


# ============================================================
# CACHE
# ============================================================

CACHE_DIR = (
    Path.home()
    / ".cache"
    / "medusa"
    / "diabetic_retinopathy"
)

STAGE1_FILE = (
    CACHE_DIR
    / STAGE1_FILENAME
)

STAGE2_FILE = (
    CACHE_DIR
    / STAGE2_FILENAME
)

THRESHOLD_FILE = (
    CACHE_DIR
    / THRESHOLD_FILENAME
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
# STAGE-SPECIFIC IMAGE TRANSFORMS
# ============================================================

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


# Stage 1 was trained at 300 × 300
STAGE1_TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    NORMALIZE,
])


# Stage 2 V2 was trained at 380 × 380
STAGE2_TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    NORMALIZE,
])


# ============================================================
# MODEL CACHE
# ============================================================

_stage1_model = None
_stage2_model = None
_stage2_threshold = 0.590


# ============================================================
# HUGGING FACE DOWNLOAD
# ============================================================

def _download_file(
    filename,
    destination,
):
    """
    Download a file from Hugging Face.
    """

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Already downloaded
    # --------------------------------------------------------

    if destination.exists():

        if destination.stat().st_size > 0:

            return destination

        destination.unlink()

    try:

        from huggingface_hub import hf_hub_download

    except ImportError as e:

        raise ImportError(
            "huggingface_hub is required for "
            "the MEDUSA diabetic retinopathy model."
        ) from e

    print(
        f"Downloading MEDUSA DR file: {filename}"
    )

    downloaded_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
    )

    # --------------------------------------------------------
    # Copy into MEDUSA cache
    # --------------------------------------------------------

    downloaded_path = Path(downloaded_path)

    if downloaded_path.resolve() != destination.resolve():

        import shutil

        shutil.copy2(
            downloaded_path,
            destination,
        )

    print(
        f"Downloaded successfully: {filename}"
    )

    return destination


# ============================================================
# ENSURE REQUIRED FILES
# ============================================================

def _ensure_files():

    _download_file(
        STAGE1_FILENAME,
        STAGE1_FILE,
    )

    _download_file(
        STAGE2_FILENAME,
        STAGE2_FILE,
    )

    _download_file(
        THRESHOLD_FILENAME,
        THRESHOLD_FILE,
    )


# ============================================================
# LOAD THRESHOLD
# ============================================================

def _load_threshold():

    global _stage2_threshold

    _stage2_threshold = 0.590

    if not THRESHOLD_FILE.exists():

        return

    try:

        with open(
            THRESHOLD_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        # ----------------------------------------------------
        # Expected:
        # {"threshold": 0.59, ...}
        # ----------------------------------------------------

        if "threshold" in data:

            _stage2_threshold = float(
                data["threshold"]
            )

    except Exception as e:

        print(
            "Warning: Could not read Stage 2 "
            f"threshold JSON: {e}"
        )

        _stage2_threshold = 0.590


# ============================================================
# CHECKPOINT STATE DICT
# ============================================================

def _get_state_dict(checkpoint):

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Invalid MEDUSA DR checkpoint format."
        )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    elif "model" in checkpoint:

        state_dict = checkpoint[
            "model"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove common prefixes
    # --------------------------------------------------------

    cleaned = {}

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

        cleaned[new_key] = value

    return cleaned


# ============================================================
# LOAD STAGE 1
# ============================================================

def _load_stage1():

    checkpoint = torch.load(
        STAGE1_FILE,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = _get_state_dict(
        checkpoint
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Stage 1 was trained with TIMM
    # --------------------------------------------------------

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=2,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# LOAD STAGE 2
# ============================================================

def _load_stage2():

    checkpoint = torch.load(
        STAGE2_FILE,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = _get_state_dict(
        checkpoint
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Stage 2 was trained with TORCHVISION
    # --------------------------------------------------------

    model = efficientnet_b3(
        weights=None,
        num_classes=2,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# LOAD BOTH MODELS
# ============================================================

def load_model():

    global _stage1_model
    global _stage2_model

    # --------------------------------------------------------
    # Already loaded
    # --------------------------------------------------------

    if (
        _stage1_model is not None
        and _stage2_model is not None
    ):

        return {
            "stage1": _stage1_model,
            "stage2": _stage2_model,
        }

    print(
        "Loading MEDUSA hierarchical "
        "diabetic retinopathy models..."
    )

    # --------------------------------------------------------
    # Download files
    # --------------------------------------------------------

    _ensure_files()

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    _load_threshold()

    print(
        f"Stage 2 threshold: "
        f"{_stage2_threshold:.3f}"
    )

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    print(
        "Loading Stage 1 "
        "(timm EfficientNet-B3)..."
    )

    _stage1_model = _load_stage1()

    print(
        "✓ Stage 1 loaded"
    )

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    print(
        "Loading Stage 2 "
        "(torchvision EfficientNet-B3)..."
    )

    _stage2_model = _load_stage2()

    print(
        "✓ Stage 2 loaded"
    )

    print(
        "✓ MEDUSA hierarchical DR model "
        "loaded successfully."
    )

    return {
        "stage1": _stage1_model,
        "stage2": _stage2_model,
    }


# ============================================================
# PREDICT
# ============================================================

def predict(image):

    if image is None:

        raise ValueError(
            "No retinal fundus image was provided."
        )

    models = load_model()

    stage1_model = models["stage1"]
    stage2_model = models["stage2"]

    # --------------------------------------------------------
    # Convert image
    # --------------------------------------------------------

    if not isinstance(
        image,
        Image.Image,
    ):

        image = Image.open(image)

    image = image.convert("RGB")

    # ========================================================
    # STAGE 1
    # ========================================================

    stage1_tensor = STAGE1_TRANSFORM(
        image
    )

    stage1_tensor = stage1_tensor.unsqueeze(
        0
    )

    stage1_tensor = stage1_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        if DEVICE.type == "cuda":

            with torch.amp.autocast(
                device_type="cuda"
            ):

                stage1_outputs = (
                    stage1_model(
                        stage1_tensor
                    )
                )

        else:

            stage1_outputs = stage1_model(
                stage1_tensor
            )

    stage1_probabilities = F.softmax(
        stage1_outputs,
        dim=1,
    )

    stage1_no_dr_probability = float(
        stage1_probabilities[0, 0].item()
    )

    stage1_dr_probability = float(
        stage1_probabilities[0, 1].item()
    )

    stage1_prediction = int(
        stage1_probabilities.argmax(
            dim=1
        ).item()
    )

    # ========================================================
    # STAGE 1 → NO DR
    # ========================================================

    if stage1_prediction == 0:

        final_class = 0

        final_confidence = (
            stage1_no_dr_probability
        )

        return {

            "prediction":
                CLASS_NAMES[final_class],

            "confidence":
                float(final_confidence),

            "probabilities": {

                "No DR":
                    round(
                        stage1_no_dr_probability,
                        6,
                    ),

                "Mild + Moderate NPDR":
                    0.0,

                "Severe NPDR + PDR":
                    0.0,
            },

            "stage1_prediction":
                "No DR",

            "stage1_confidence":
                round(
                    stage1_no_dr_probability,
                    6,
                ),

            "stage1_no_dr_probability":
                round(
                    stage1_no_dr_probability,
                    6,
                ),

            "stage1_dr_probability":
                round(
                    stage1_dr_probability,
                    6,
                ),

            "stage2_prediction":
                None,

            "stage2_threshold":
                float(
                    _stage2_threshold
                ),

            "screening":
                True,
        }

    # ========================================================
    # STAGE 1 → ANY DR
    # ========================================================

    # Stage 2 uses 380 × 380
    stage2_tensor = STAGE2_TRANSFORM(
        image
    )

    stage2_tensor = stage2_tensor.unsqueeze(
        0
    )

    stage2_tensor = stage2_tensor.to(
        DEVICE
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    with torch.no_grad():

        if DEVICE.type == "cuda":

            with torch.amp.autocast(
                device_type="cuda"
            ):

                stage2_outputs = (
                    stage2_model(
                        stage2_tensor
                    )
                )

        else:

            stage2_outputs = stage2_model(
                stage2_tensor
            )

    stage2_probabilities = F.softmax(
        stage2_outputs,
        dim=1,
    )

    mild_moderate_probability = float(
        stage2_probabilities[0, 0].item()
    )

    severe_probability = float(
        stage2_probabilities[0, 1].item()
    )

    # ========================================================
    # LOCKED THRESHOLD
    # ========================================================

    if (
        severe_probability
        >= _stage2_threshold
    ):

        final_class = 2

        final_confidence = (
            severe_probability
        )

        stage2_prediction = (
            "Severe/PDR"
        )

    else:

        final_class = 1

        final_confidence = (
            mild_moderate_probability
        )

        stage2_prediction = (
            "Mild/Moderate"
        )

    # ========================================================
    # FINAL PROBABILITY DICTIONARY
    # ========================================================

    probability_dict = {

        "No DR":
            round(
                stage1_no_dr_probability,
                6,
            ),

        "Mild + Moderate NPDR":
            round(
                mild_moderate_probability,
                6,
            ),

        "Severe NPDR + PDR":
            round(
                severe_probability,
                6,
            ),
    }

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "prediction":
            CLASS_NAMES[final_class],

        "confidence":
            float(final_confidence),

        "probabilities":
            probability_dict,

        # ----------------------------------------------------
        # Stage 1 information
        # ----------------------------------------------------

        "stage1_prediction":
            "Any DR",

        "stage1_confidence":
            round(
                stage1_dr_probability,
                6,
            ),

        "stage1_no_dr_probability":
            round(
                stage1_no_dr_probability,
                6,
            ),

        "stage1_dr_probability":
            round(
                stage1_dr_probability,
                6,
            ),

        # ----------------------------------------------------
        # Stage 2 information
        # ----------------------------------------------------

        "stage2_prediction":
            stage2_prediction,

        "stage2_mild_moderate_probability":
            round(
                mild_moderate_probability,
                6,
            ),

        "stage2_severe_probability":
            round(
                severe_probability,
                6,
            ),

        "stage2_threshold":
            float(
                _stage2_threshold
            ),

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        "screening":
            True,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_info():

    return {

        "model":
            "Hierarchical EfficientNet-B3",

        "architecture":
            "2-stage EfficientNet-B3",

        "stage1_architecture":
            "timm efficientnet_b3",

        "stage2_architecture":
            "torchvision efficientnet_b3",

        "source":
            "Hugging Face",

        "repository":
            HF_REPO,

        "stage1_model_file":
            STAGE1_FILENAME,

        "stage2_model_file":
            STAGE2_FILENAME,

        "threshold_file":
            THRESHOLD_FILENAME,

        "num_classes":
            3,

        "classes":
            CLASS_NAMES,

        "stage1_input_size":
            [300, 300],

        "stage2_input_size":
            [380, 380],

        "stage2_threshold":
            float(
                _stage2_threshold
            ),

        "device":
            str(DEVICE),

        "type":
            "screening / decision support",
    }
