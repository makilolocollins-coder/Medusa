import streamlit as st
import torch
import timm

from PIL import Image

from torchvision import transforms

from huggingface_hub import (
    HfApi,
    hf_hub_download,
)


# ============================================================
# CONFIG
# ============================================================

HF_REPO = (
    "Makky07/MammoSense-breast-ultrasound"
)

IMAGE_SIZE = 224


# ============================================================
# FIND MODEL
# ============================================================

@st.cache_data(ttl=3600)
def find_model_file():

    api = HfApi()

    files = api.list_repo_files(
        repo_id=HF_REPO,
        repo_type="model",
    )

    model_files = [
        file
        for file in files
        if file.lower().endswith(
            (
                ".pt",
                ".pth",
                ".bin",
            )
        )
    ]

    if not model_files:

        raise RuntimeError(
            "No PyTorch model was found "
            "in the Hugging Face repository."
        )


    # Prefer GAIA/MammoSense files

    preferred = [
        file
        for file in model_files
        if (
            "gaia_busi" in file.lower()
            or "mammosense" in file.lower()
        )
    ]


    if preferred:

        return preferred[0]


    return model_files[0]


# ============================================================
# LOAD MODEL
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
    # FIND MODEL
    # --------------------------------------------------------

    model_file = (
        find_model_file()
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=model_file,
        repo_type="model",
    )


    # --------------------------------------------------------
    # PYTORCH 2.6
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

    except Exception as e:

        raise RuntimeError(
            "Unable to load MammoSense checkpoint:\n\n"
            + str(e)
        )


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "The checkpoint is not a valid "
            "MammoSense checkpoint."
        )


    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    architecture = checkpoint.get(
        "architecture",
        "vit_small_patch16_224",
    )


    # --------------------------------------------------------
    # NUMBER OF CLASSES
    # --------------------------------------------------------

    num_classes = checkpoint.get(
        "num_classes",
        3,
    )


    try:

        num_classes = int(
            num_classes
        )

    except Exception:

        num_classes = 3


    # --------------------------------------------------------
    # CLASS NAMES
    # --------------------------------------------------------

    class_names = checkpoint.get(
        "class_names",
        None,
    )


    if (
        isinstance(
            class_names,
            (list, tuple),
        )
        and len(class_names) == num_classes
    ):

        classes = list(
            class_names
        )

    else:

        default_classes = [
            "Normal",
            "Benign",
            "Malignant",
        ]

        classes = default_classes[
            :num_classes
        ]


    # --------------------------------------------------------
    # STATE DICT
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = {
            key: value
            for key, value in checkpoint.items()
            if torch.is_tensor(value)
        }


    if not state_dict:

        raise RuntimeError(
            "No model weights were found "
            "inside the checkpoint."
        )


    # --------------------------------------------------------
    # CLEAN PREFIXES
    # --------------------------------------------------------

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                7:
            ]

        if new_key.startswith(
            "model."
        ):

            new_key = new_key[
                6:
            ]

        cleaned[
            new_key
        ] = value


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    try:

        model = timm.create_model(
            architecture,
            pretrained=False,
            num_classes=num_classes,
            img_size=IMAGE_SIZE,
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not create architecture "
            f"{architecture}:\n\n{e}"
        )


    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            cleaned,
            strict=True,
        )

    except RuntimeError as e:

        expected = set(
            model.state_dict().keys()
        )

        actual = set(
            cleaned.keys()
        )

        missing = list(
            expected - actual
        )[:20]

        unexpected = list(
            actual - expected
        )[:20]

        raise RuntimeError(
            "MammoSense checkpoint does not "
            "match the expected architecture.\n\n"

            f"Architecture: {architecture}\n"
            f"Classes: {classes}\n\n"

            f"Missing keys:\n{missing}\n\n"

            f"Unexpected keys:\n{unexpected}\n\n"

            f"Original error:\n{e}"
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


    return {
        "model": model,
        "transform": transform,
        "classes": classes,
        "device": device,
        "model_file": model_file,
        "architecture": architecture,
    }


# ============================================================
# PREDICTION
# ============================================================

@torch.inference_mode()
def predict(image):

    package = (
        load_model()
    )

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
            probabilities[
                i
            ].item()
        )


    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_dict,
    }
