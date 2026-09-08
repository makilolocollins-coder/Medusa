# ============================================================
# MEDUSA AI
# DIABETIC RETINOPATHY
#
# Hierarchical 2-Stage EfficientNet-B3
#
# Stage 1:
#   No DR vs Any DR
#   Architecture: torchvision EfficientNet-B3
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
import shutil

import torch
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b3


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO = "Makky07/Retinopathy"


# ------------------------------------------------------------
# NEW MEDUSA MODEL FILES
# ------------------------------------------------------------

STAGE1_FILENAME = "medusa_stage1.pt"
STAGE1_JSON_FILENAME = "medusa_stage1.json"

STAGE2_FILENAME = "medusa_stage2.pt"
STAGE2_JSON_FILENAME = "medusa_stage2.json"


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

STAGE1_JSON_FILE = (
    CACHE_DIR
    / STAGE1_JSON_FILENAME
)


STAGE2_FILE = (
    CACHE_DIR
    / STAGE2_FILENAME
)

STAGE2_JSON_FILE = (
    CACHE_DIR
    / STAGE2_JSON_FILENAME
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
# NORMALIZATION
# ============================================================

NORMALIZE = transforms.Normalize(
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
)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================
#
# IMPORTANT:
#
# The new training code uses 300 × 300 for BOTH stages.
#
# Therefore Stage 2 is no longer 380 × 380.
# ============================================================

STAGE1_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (300, 300)
    ),

    transforms.ToTensor(),

    NORMALIZE,
])


STAGE2_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (300, 300)
    ),

    transforms.ToTensor(),

    NORMALIZE,
])


# ============================================================
# MODEL CACHE
# ============================================================

_stage1_model = None

_stage2_model = None

_stage2_threshold = 0.50


# ============================================================
# HUGGING FACE DOWNLOAD
# ============================================================

def _download_file(
    filename,
    destination,
):
    """
    Download a MEDUSA model file from Hugging Face.
    """

    destination = Path(
        destination
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Already cached
    # --------------------------------------------------------

    if destination.exists():

        if destination.stat().st_size > 0:

            return destination

        destination.unlink()


    # --------------------------------------------------------
    # Import Hugging Face Hub
    # --------------------------------------------------------

    try:

        from huggingface_hub import (
            hf_hub_download
        )

    except ImportError as e:

        raise ImportError(
            "huggingface_hub is required "
            "for MEDUSA diabetic retinopathy."
        ) from e


    print(
        f"Downloading MEDUSA DR file: "
        f"{filename}"
    )


    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    downloaded_path = hf_hub_download(

        repo_id=HF_REPO,

        filename=filename,
    )


    downloaded_path = Path(
        downloaded_path
    )


    # --------------------------------------------------------
    # Copy to MEDUSA cache
    # --------------------------------------------------------

    if (
        downloaded_path.resolve()
        != destination.resolve()
    ):

        shutil.copy2(
            downloaded_path,
            destination,
        )


    print(
        f"✓ Downloaded: {filename}"
    )


    return destination


# ============================================================
# ENSURE REQUIRED FILES
# ============================================================

def _ensure_files():

    # Stage 1 model
    _download_file(
        STAGE1_FILENAME,
        STAGE1_FILE,
    )


    # Stage 1 metadata
    _download_file(
        STAGE1_JSON_FILENAME,
        STAGE1_JSON_FILE,
    )


    # Stage 2 model
    _download_file(
        STAGE2_FILENAME,
        STAGE2_FILE,
    )


    # Stage 2 metadata
    _download_file(
        STAGE2_JSON_FILENAME,
        STAGE2_JSON_FILE,
    )


# ============================================================
# LOAD JSON METADATA
# ============================================================

def _load_json(
    filepath
):

    filepath = Path(
        filepath
    )


    if not filepath.exists():

        return {}


    try:

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)


    except Exception as e:

        print(
            f"Warning: Could not read "
            f"{filepath.name}: {e}"
        )

        return {}


# ============================================================
# LOAD STAGE 2 THRESHOLD
# ============================================================

def _load_threshold():

    global _stage2_threshold


    # --------------------------------------------------------
    # Safe default
    # --------------------------------------------------------

    _stage2_threshold = 0.50


    metadata = _load_json(
        STAGE2_JSON_FILE
    )


    # --------------------------------------------------------
    # New training JSON
    # --------------------------------------------------------

    if (
        "selected_threshold"
        in metadata
    ):

        try:

            _stage2_threshold = float(
                metadata[
                    "selected_threshold"
                ]
            )

            return

        except Exception:

            pass


    # --------------------------------------------------------
    # Fallback if older metadata format
    # --------------------------------------------------------

    if (
        "threshold"
        in metadata
    ):

        try:

            _stage2_threshold = float(
                metadata[
                    "threshold"
                ]
            )

            return

        except Exception:

            pass


    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    print(
        "Warning: Stage 2 threshold was "
        "not found in JSON."
    )

    print(
        "Using default threshold: 0.500"
    )


# ============================================================
# CHECKPOINT STATE DICT
# ============================================================

def _get_state_dict(
    checkpoint
):

    # --------------------------------------------------------
    # Checkpoint must be dictionary
    # --------------------------------------------------------

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Invalid MEDUSA checkpoint format."
        )


    # --------------------------------------------------------
    # New checkpoints
    #
    # The new training code saves:
    #
    # torch.save(clean_state, output_pt)
    #
    # Therefore the checkpoint itself is the state dict.
    # --------------------------------------------------------

    if (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        "state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    elif (
        "model"
        in checkpoint
        and isinstance(
            checkpoint["model"],
            dict,
        )
    ):

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


        cleaned[
            new_key
        ] = value


    return cleaned


# ============================================================
# LOAD STAGE 1
# ============================================================

def _load_stage1():

    print(
        "Loading Stage 1 "
        "(torchvision EfficientNet-B3)..."
    )


    checkpoint = torch.load(

        STAGE1_FILE,

        map_location="cpu",

        weights_only=True,
    )


    state_dict = _get_state_dict(
        checkpoint
    )


    # --------------------------------------------------------
    # NEW MODEL
    #
    # Training:
    #
    # efficientnet_b3(
    #     weights=DEFAULT
    # )
    #
    # classifier -> Linear(..., 2)
    # --------------------------------------------------------

    model = efficientnet_b3(

        weights=None,

        num_classes=2,
    )


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )


    if missing:

        raise RuntimeError(
            "Stage 1 checkpoint is missing "
            f"model parameters:\n{missing}"
        )


    if unexpected:

        print(
            "Warning: unexpected Stage 1 "
            f"parameters: {unexpected}"
        )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )


    model.eval()


    print(
        "✓ Stage 1 loaded"
    )


    return model


# ============================================================
# LOAD STAGE 2
# ============================================================

def _load_stage2():

    print(
        "Loading Stage 2 "
        "(torchvision EfficientNet-B3)..."
    )


    checkpoint = torch.load(

        STAGE2_FILE,

        map_location="cpu",

        weights_only=True,
    )


    state_dict = _get_state_dict(
        checkpoint
    )


    # --------------------------------------------------------
    # NEW MODEL
    # --------------------------------------------------------

    model = efficientnet_b3(

        weights=None,

        num_classes=2,
    )


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )


    if missing:

        raise RuntimeError(
            "Stage 2 checkpoint is missing "
            f"model parameters:\n{missing}"
        )


    if unexpected:

        print(
            "Warning: unexpected Stage 2 "
            f"parameters: {unexpected}"
        )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )


    model.eval()


    print(
        "✓ Stage 2 loaded"
    )


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
        and
        _stage2_model is not None
    ):

        return {

            "stage1":
                _stage1_model,

            "stage2":
                _stage2_model,
        }


    print(
        "Loading MEDUSA hierarchical "
        "diabetic retinopathy models..."
    )


    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    _ensure_files()


    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    _load_threshold()


    print(
        f"Stage 2 threshold: "
        f"{_stage2_threshold:.3f}"
    )


    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    _stage1_model = (
        _load_stage1()
    )


    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    _stage2_model = (
        _load_stage2()
    )


    print(
        "✓ MEDUSA hierarchical DR "
        "model loaded successfully."
    )


    return {

        "stage1":
            _stage1_model,

        "stage2":
            _stage2_model,
    }


# ============================================================
# PREDICT
# ============================================================

def predict(
    image
):

    if image is None:

        raise ValueError(
            "No retinal fundus image "
            "was provided."
        )


    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models = load_model()


    stage1_model = (
        models["stage1"]
    )

    stage2_model = (
        models["stage2"]
    )


    # ========================================================
    # CONVERT IMAGE
    # ========================================================

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


    # ========================================================
    # STAGE 1
    # ========================================================

    stage1_tensor = (
        STAGE1_TRANSFORM(
            image
        )
    )


    stage1_tensor = (
        stage1_tensor.unsqueeze(0)
    )


    stage1_tensor = (
        stage1_tensor.to(
            DEVICE
        )
    )


    # --------------------------------------------------------
    # Stage 1 inference
    # --------------------------------------------------------

    with torch.no_grad():

        if DEVICE.type == "cuda":

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                stage1_outputs = (
                    stage1_model(
                        stage1_tensor
                    )
                )

        else:

            stage1_outputs = (
                stage1_model(
                    stage1_tensor
                )
            )


    stage1_probabilities = (
        F.softmax(
            stage1_outputs,
            dim=1,
        )
    )


    # --------------------------------------------------------
    # Stage 1 probabilities
    #
    # 0 = No DR
    # 1 = Any DR
    # --------------------------------------------------------

    stage1_no_dr_probability = float(
        stage1_probabilities[
            0,
            0
        ].item()
    )


    stage1_dr_probability = float(
        stage1_probabilities[
            0,
            1
        ].item()
    )


    stage1_prediction = int(
        stage1_probabilities
        .argmax(
            dim=1
        )
        .item()
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
                CLASS_NAMES[
                    final_class
                ],


            "confidence":
                float(
                    final_confidence
                ),


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


            # ------------------------------------------------
            # Stage 1
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Stage 2
            # ------------------------------------------------

            "stage2_prediction":
                None,


            "stage2_mild_moderate_probability":
                None,


            "stage2_severe_probability":
                None,


            "stage2_threshold":
                float(
                    _stage2_threshold
                ),


            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            "screening":
                True,
        }


    # ========================================================
    # STAGE 1 → ANY DR
    # ========================================================

    stage2_tensor = (
        STAGE2_TRANSFORM(
            image
        )
    )


    stage2_tensor = (
        stage2_tensor.unsqueeze(0)
    )


    stage2_tensor = (
        stage2_tensor.to(
            DEVICE
        )
    )


    # ========================================================
    # STAGE 2
    # ========================================================

    with torch.no_grad():

        if DEVICE.type == "cuda":

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                stage2_outputs = (
                    stage2_model(
                        stage2_tensor
                    )
                )

        else:

            stage2_outputs = (
                stage2_model(
                    stage2_tensor
                )
            )


    stage2_probabilities = (
        F.softmax(
            stage2_outputs,
            dim=1,
        )
    )


    # --------------------------------------------------------
    # Stage 2 probabilities
    #
    # 0 = Mild/Moderate
    # 1 = Severe/PDR
    # --------------------------------------------------------

    mild_moderate_probability = float(
        stage2_probabilities[
            0,
            0
        ].item()
    )


    severe_probability = float(
        stage2_probabilities[
            0,
            1
        ].item()
    )


    # ========================================================
    # STAGE 2 THRESHOLD
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
    # FINAL RESULT
    # ========================================================

    return {

        "prediction":
            CLASS_NAMES[
                final_class
            ],


        "confidence":
            float(
                final_confidence
            ),


        "probabilities": {

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
        },


        # ====================================================
        # STAGE 1
        # ====================================================

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


        # ====================================================
        # STAGE 2
        # ====================================================

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


        # ====================================================
        # METADATA
        # ====================================================

        "screening":
            True,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_info():

    return {

        "model":
            "MEDUSA Hierarchical EfficientNet-B3",


        "architecture":
            "2-stage EfficientNet-B3",


        "stage1_architecture":
            "torchvision EfficientNet-B3",


        "stage2_architecture":
            "torchvision EfficientNet-B3",


        "source":
            "Hugging Face",


        "repository":
            HF_REPO,


        # ----------------------------------------------------
        # Files
        # ----------------------------------------------------

        "stage1_model_file":
            STAGE1_FILENAME,


        "stage1_metadata_file":
            STAGE1_JSON_FILENAME,


        "stage2_model_file":
            STAGE2_FILENAME,


        "stage2_metadata_file":
            STAGE2_JSON_FILENAME,


        # ----------------------------------------------------
        # Classes
        # ----------------------------------------------------

        "num_classes":
            3,


        "classes":
            CLASS_NAMES,


        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        "stage1_input_size":
            [300, 300],


        "stage2_input_size":
            [300, 300],


        # ----------------------------------------------------
        # Stage 2 threshold
        # ----------------------------------------------------

        "stage2_threshold":
            float(
                _stage2_threshold
            ),


        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        "device":
            str(
                DEVICE
            ),


        # ----------------------------------------------------
        # Type
        # ----------------------------------------------------

        "type":
            "screening / decision support",
    }
