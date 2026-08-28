# ================================================================
# MAMMOSENSE BRAIN V3.1
# 3D BRAIN TUMOR SEGMENTATION INFERENCE
#
# Hugging Face:
# Makky07/Brain_Tumor
#
# Checkpoint:
# mammosense_brain_v31_best.pt
#
# INPUT:
#   T1
#   T1CE
#   T2
#   FLAIR
#
# INPUT SHAPE:
#   [1, 4, 96, 96, 96]
#
# OUTPUT:
#   Binary tumor segmentation
#
# IMPORTANT:
#   This module performs AI-assisted segmentation.
#   It does NOT provide a clinical diagnosis.
# ================================================================

import io
import os
import tempfile
import urllib.request

import numpy as np
import nibabel as nib
from PIL import Image

import streamlit as st

import torch
import torch.nn as nn

from scipy.ndimage import zoom, label


# ================================================================
# CONFIGURATION
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SHAPE = (
    96,
    96,
    96,
)

HF_REPO = (
    "Makky07/Brain_Tumor"
)

HF_MODEL_URL = (
    "https://huggingface.co/"
    "Makky07/Brain_Tumor/"
    "resolve/main/"
    "mammosense_brain_v31_best.pt"
)

MODEL_NAME = (
    "MammoSense Brain V3.1"
)

ARCHITECTURE = (
    "3D_U-Net"
)

INPUT_MODALITIES = [
    "T1",
    "T1CE",
    "T2",
    "FLAIR",
]


# ================================================================
# INFERENCE SAFETY THRESHOLDS
# ================================================================
#
# These are engineering safeguards used to reject extremely small
# isolated predictions.
#
# They are NOT clinical diagnostic thresholds.
#
# They should ideally be tuned against a held-out validation set
# before being used for clinical research claims.
# ================================================================

SEGMENTATION_THRESHOLD = 0.5

MIN_TUMOR_FRACTION = 0.001

MIN_CONNECTED_VOXELS = 100


# ================================================================
# 3D U-NET
#
# EXACT ARCHITECTURE USED DURING V3.1 TRAINING
# ================================================================


class DoubleConv3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.InstanceNorm3d(
                out_channels
            ),

            nn.LeakyReLU(
                0.01,
                inplace=True,
            ),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.InstanceNorm3d(
                out_channels
            ),

            nn.LeakyReLU(
                0.01,
                inplace=True,
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.block(x)


class UNet3D(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------------
        # ENCODER
        # --------------------------------------------------------

        self.enc1 = DoubleConv3D(
            4,
            32,
        )

        self.enc2 = DoubleConv3D(
            32,
            64,
        )

        self.enc3 = DoubleConv3D(
            64,
            128,
        )

        self.enc4 = DoubleConv3D(
            128,
            256,
        )

        self.pool = nn.MaxPool3d(
            kernel_size=2
        )

        # --------------------------------------------------------
        # BOTTLENECK
        # --------------------------------------------------------

        self.bottleneck = DoubleConv3D(
            256,
            512,
        )

        # --------------------------------------------------------
        # DECODER
        # --------------------------------------------------------

        self.up4 = nn.ConvTranspose3d(
            512,
            256,
            kernel_size=2,
            stride=2,
        )

        self.dec4 = DoubleConv3D(
            512,
            256,
        )

        self.up3 = nn.ConvTranspose3d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv3D(
            256,
            128,
        )

        self.up2 = nn.ConvTranspose3d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv3D(
            128,
            64,
        )

        self.up1 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv3D(
            64,
            32,
        )

        # --------------------------------------------------------
        # OUTPUT
        # --------------------------------------------------------

        self.output = nn.Conv3d(
            32,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        x,
    ):

        # --------------------------------------------------------
        # ENCODER
        # --------------------------------------------------------

        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # --------------------------------------------------------
        # BOTTLENECK
        # --------------------------------------------------------

        b = self.bottleneck(
            self.pool(e4)
        )

        # --------------------------------------------------------
        # DECODER
        # --------------------------------------------------------

        d4 = self.up4(
            b
        )

        d4 = torch.cat(
            [
                d4,
                e4,
            ],
            dim=1,
        )

        d4 = self.dec4(
            d4
        )

        d3 = self.up3(
            d4
        )

        d3 = torch.cat(
            [
                d3,
                e3,
            ],
            dim=1,
        )

        d3 = self.dec3(
            d3
        )

        d2 = self.up2(
            d3
        )

        d2 = torch.cat(
            [
                d2,
                e2,
            ],
            dim=1,
        )

        d2 = self.dec2(
            d2
        )

        d1 = self.up1(
            d2
        )

        d1 = torch.cat(
            [
                d1,
                e1,
            ],
            dim=1,
        )

        d1 = self.dec1(
            d1
        )

        return self.output(
            d1
        )


# ================================================================
# MODEL DOWNLOAD
# ================================================================


@st.cache_resource(
    show_spinner=False
)
def _download_model():

    cache_dir = os.path.join(
        tempfile.gettempdir(),
        "mammosense_brain",
    )

    os.makedirs(
        cache_dir,
        exist_ok=True,
    )

    model_path = os.path.join(
        cache_dir,
        "mammosense_brain_v31_best.pt",
    )

    if not os.path.exists(
        model_path
    ):

        try:

            urllib.request.urlretrieve(
                HF_MODEL_URL,
                model_path,
            )

        except Exception as error:

            if os.path.exists(
                model_path
            ):

                try:
                    os.remove(
                        model_path
                    )
                except Exception:
                    pass

            raise RuntimeError(
                "Unable to download the "
                "MammoSense Brain V3.1 checkpoint "
                "from Hugging Face."
            ) from error

    if not os.path.isfile(
        model_path
    ):

        raise RuntimeError(
            "Brain tumor model file "
            "was not created."
        )

    if os.path.getsize(
        model_path
    ) < 1024:

        raise RuntimeError(
            "Downloaded Brain V3.1 checkpoint "
            "appears to be invalid or incomplete."
        )

    return model_path


# ================================================================
# LOAD MODEL
# ================================================================


@st.cache_resource(
    show_spinner=False
)
def load_model():

    model_path = _download_model()

    model = UNet3D()

    try:

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
        )

    except Exception as error:

        raise RuntimeError(
            "Unable to load the "
            "MammoSense Brain V3.1 checkpoint."
        ) from error

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Invalid MammoSense Brain checkpoint. "
            "Expected a checkpoint dictionary."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )

    if not isinstance(
        state_dict,
        dict,
    ):

        raise RuntimeError(
            "Invalid model_state_dict "
            "inside Brain V3.1 checkpoint."
        )

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except Exception as error:

        raise RuntimeError(
            "Brain V3.1 checkpoint architecture "
            "does not match the expected 3D U-Net."
        ) from error

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ================================================================
# NIFTI LOADING
# ================================================================


def load_nifti(
    file_bytes,
):

    if file_bytes is None:

        raise ValueError(
            "No NIfTI file was provided."
        )

    if not file_bytes:

        raise ValueError(
            "Empty NIfTI file."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".nii.gz",
            delete=False,
        ) as tmp:

            tmp.write(
                file_bytes
            )

            temp_path = tmp.name

        nii = nib.load(
            temp_path
        )

        volume = nii.get_fdata(
            dtype=np.float32
        )

        volume = np.asarray(
            volume,
            dtype=np.float32,
        )

    except Exception as error:

        raise ValueError(
            "Unable to read the uploaded "
            "NIfTI file."
        ) from error

    finally:

        if temp_path is not None:

            try:

                os.remove(
                    temp_path
                )

            except Exception:
                pass

    if volume.ndim != 3:

        raise ValueError(
            "Expected a 3D NIfTI volume. "
            f"Received shape {volume.shape}."
        )

    if not np.isfinite(
        volume
    ).any():

        raise ValueError(
            "NIfTI volume contains no "
            "valid numerical values."
        )

    return volume


# ================================================================
# MRI NORMALIZATION
#
# EXACT NORMALIZATION USED DURING TRAINING
# ================================================================


def normalize_mri(
    volume,
):

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    nonzero = volume[
        volume != 0
    ]

    if nonzero.size == 0:

        return np.zeros_like(
            volume,
            dtype=np.float32,
        )

    mean = np.mean(
        nonzero
    )

    std = np.std(
        nonzero
    )

    if std < 1e-6:

        return np.zeros_like(
            volume,
            dtype=np.float32,
        )

    volume = (
        volume - mean
    ) / std

    volume = np.clip(
        volume,
        -5.0,
        5.0,
    )

    return volume.astype(
        np.float32
    )


# ================================================================
# RESIZE MRI
#
# EXACT RESAMPLING USED DURING TRAINING
# ================================================================


def resize_volume(
    volume,
    target_shape=IMAGE_SHAPE,
):

    if volume.ndim != 3:

        raise ValueError(
            "MRI volume must be 3-dimensional."
        )

    if any(
        dimension <= 0
        for dimension in volume.shape
    ):

        raise ValueError(
            "MRI volume has an invalid shape."
        )

    factors = [
        target_shape[i]
        / volume.shape[i]
        for i in range(3)
    ]

    resized = zoom(
        volume,
        factors,
        order=1,
    )

    resized = np.asarray(
        resized,
        dtype=np.float32,
    )

    if resized.shape != target_shape:

        resized = zoom(
            resized,
            [
                target_shape[i]
                / resized.shape[i]
                for i in range(3)
            ],
            order=1,
        )

    return resized.astype(
        np.float32
    )


# ================================================================
# PREPARE FOUR MRI MODALITIES
# ================================================================


def prepare_volume(
    t1_bytes,
    t1ce_bytes,
    t2_bytes,
    flair_bytes,
):

    files = [
        (
            "T1",
            t1_bytes,
        ),
        (
            "T1CE",
            t1ce_bytes,
        ),
        (
            "T2",
            t2_bytes,
        ),
        (
            "FLAIR",
            flair_bytes,
        ),
    ]

    channels = []

    original_shapes = {}

    for name, file_bytes in files:

        if file_bytes is None:

            raise ValueError(
                f"{name} MRI file is required."
            )

        volume = load_nifti(
            file_bytes
        )

        original_shapes[
            name
        ] = list(
            volume.shape
        )

        volume = normalize_mri(
            volume
        )

        volume = resize_volume(
            volume,
            IMAGE_SHAPE,
        )

        channels.append(
            volume
        )

    image = np.stack(
        channels,
        axis=0,
    )

    if image.shape != (
        4,
        96,
        96,
        96,
    ):

        raise RuntimeError(
            "Prepared MRI tensor has an "
            "unexpected shape: "
            f"{image.shape}"
        )

    image = torch.from_numpy(
        image
    ).float()

    image = image.unsqueeze(
        0
    )

    return (
        image,
        original_shapes,
    )


# ================================================================
# CONNECTED COMPONENT ANALYSIS
# ================================================================


def analyze_segmentation(
    mask,
):

    mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    tumor_voxels = int(
        mask.sum()
    )

    total_voxels = int(
        mask.size
    )

    tumor_fraction = (
        tumor_voxels
        / max(
            total_voxels,
            1,
        )
    )

    tumor_percentage = (
        tumor_fraction
        * 100.0
    )

    # ------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------

    connected_mask, number_of_regions = label(
        mask
    )

    if number_of_regions > 0:

        region_sizes = np.bincount(
            connected_mask.ravel()
        )

        if region_sizes.size > 0:

            region_sizes[0] = 0

        largest_region_voxels = int(
            region_sizes.max()
        )

    else:

        largest_region_voxels = 0

    # ------------------------------------------------------------
    # Detection decision
    # ------------------------------------------------------------

    tumor_detected = (
        tumor_fraction
        >= MIN_TUMOR_FRACTION
        and
        largest_region_voxels
        >= MIN_CONNECTED_VOXELS
    )

    return {
        "tumor_voxels": tumor_voxels,

        "total_voxels": total_voxels,

        "tumor_fraction": (
            tumor_fraction
        ),

        "tumor_percentage": (
            tumor_percentage
        ),

        "number_of_regions": (
            int(number_of_regions)
        ),

        "largest_region_voxels": (
            largest_region_voxels
        ),

        "tumor_detected": (
            bool(tumor_detected)
        ),
    }


# ================================================================
# PREDICTION
# ================================================================


@torch.no_grad()
def predict(
    t1_bytes,
    t1ce_bytes,
    t2_bytes,
    flair_bytes,
):

    model = load_model()

    image, original_shapes = (
        prepare_volume(
            t1_bytes,
            t1ce_bytes,
            t2_bytes,
            flair_bytes,
        )
    )

    image = image.to(
        DEVICE,
        non_blocking=True,
    )

    # ------------------------------------------------------------
    # MODEL INFERENCE
    # ------------------------------------------------------------

    if DEVICE.type == "cuda":

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            logits = model(
                image
            )

    else:

        logits = model(
            image
        )

    # ------------------------------------------------------------
    # PROBABILITY
    # ------------------------------------------------------------

    probabilities = torch.sigmoid(
        logits
    )

    probability_volume = (
        probabilities[
            0,
            0,
        ]
        .float()
        .cpu()
        .numpy()
    )

    probability_volume = np.nan_to_num(
        probability_volume,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    probability_volume = np.clip(
        probability_volume,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------
    # BINARY MASK
    # ------------------------------------------------------------

    mask = (
        probability_volume
        >= SEGMENTATION_THRESHOLD
    ).astype(
        np.uint8
    )

    # ------------------------------------------------------------
    # SEGMENTATION ANALYSIS
    # ------------------------------------------------------------

    segmentation = (
        analyze_segmentation(
            mask
        )
    )

    tumor_voxels = (
        segmentation[
            "tumor_voxels"
        ]
    )

    total_voxels = (
        segmentation[
            "total_voxels"
        ]
    )

    tumor_fraction = (
        segmentation[
            "tumor_fraction"
        ]
    )

    tumor_percentage = (
        segmentation[
            "tumor_percentage"
        ]
    )

    number_of_regions = (
        segmentation[
            "number_of_regions"
        ]
    )

    largest_region_voxels = (
        segmentation[
            "largest_region_voxels"
        ]
    )

    tumor_detected = (
        segmentation[
            "tumor_detected"
        ]
    )

    # ------------------------------------------------------------
    # PROBABILITY STATISTICS
    # ------------------------------------------------------------

    positive_probabilities = (
        probability_volume[
            mask == 1
        ]
    )

    if positive_probabilities.size > 0:

        confidence = float(
            positive_probabilities.mean()
        )

    else:

        confidence = float(
            1.0
            - probability_volume.mean()
        )

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

    maximum_probability = float(
        probability_volume.max()
    )

    mean_probability = float(
        probability_volume.mean()
    )

    # ------------------------------------------------------------
    # HUMAN-READABLE RESULT
    # ------------------------------------------------------------

    if tumor_detected:

        prediction = (
            "TUMOR DETECTED"
        )

    else:

        prediction = (
            "NO SIGNIFICANT "
            "TUMOR SEGMENTATION"
        )

    # ------------------------------------------------------------
    # RETURN
    # ------------------------------------------------------------

    return {

        # --------------------------------------------------------
        # PRIMARY RESULT
        # --------------------------------------------------------

        "prediction": prediction,

        "confidence": confidence,

        "confidence_percentage": (
            confidence * 100.0
        ),

        "tumor_detected": (
            bool(tumor_detected)
        ),

        # --------------------------------------------------------
        # SEGMENTATION
        # --------------------------------------------------------

        "tumor_voxels": (
            tumor_voxels
        ),

        "total_voxels": (
            total_voxels
        ),

        "tumor_fraction": (
            tumor_fraction
        ),

        "tumor_percentage": (
            tumor_percentage
        ),

        "number_of_regions": (
            number_of_regions
        ),

        "largest_region_voxels": (
            largest_region_voxels
        ),

        # --------------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------------

        "maximum_probability": (
            maximum_probability
        ),

        "mean_probability": (
            mean_probability
        ),

        # --------------------------------------------------------
        # RAW OUTPUTS
        # --------------------------------------------------------

        "mask": mask,

        "probability_volume": (
            probability_volume
        ),

        # --------------------------------------------------------
        # INPUT INFORMATION
        # --------------------------------------------------------

        "original_shapes": (
            original_shapes
        ),

        "input_shape": list(
            IMAGE_SHAPE
        ),

        "input_modalities": (
            INPUT_MODALITIES.copy()
        ),

        # --------------------------------------------------------
        # MODEL INFORMATION
        # --------------------------------------------------------

        "model": MODEL_NAME,

        "architecture": ARCHITECTURE,

        "dataset": (
            "BraTS 2021"
        ),

        "huggingface_repo": (
            HF_REPO
        ),

        # --------------------------------------------------------
        # THRESHOLDS
        # --------------------------------------------------------

        "segmentation_threshold": (
            SEGMENTATION_THRESHOLD
        ),

        "minimum_tumor_fraction": (
            MIN_TUMOR_FRACTION
        ),

        "minimum_connected_voxels": (
            MIN_CONNECTED_VOXELS
        ),

        # --------------------------------------------------------
        # HARDWARE
        # --------------------------------------------------------

        "device": str(
            DEVICE
        ),
    }


# ================================================================
# REPRESENTATIVE MRI SLICE + TUMOR OVERLAY
# ================================================================


def create_overlay(
    flair_bytes,
    mask,
):

    if mask is None:

        raise ValueError(
            "Segmentation mask is required."
        )

    mask = np.asarray(
        mask
    )

    if mask.ndim != 3:

        raise ValueError(
            "Segmentation mask must be 3-dimensional."
        )

    flair = load_nifti(
        flair_bytes
    )

    flair = normalize_mri(
        flair
    )

    flair = resize_volume(
        flair,
        IMAGE_SHAPE,
    )

    mask = resize_mask_for_overlay(
        mask,
        IMAGE_SHAPE,
    )

    # ------------------------------------------------------------
    # Middle axial slice
    # ------------------------------------------------------------

    slice_index = (
        flair.shape[2] // 2
    )

    image_slice = flair[
        :,
        :,
        slice_index,
    ]

    mask_slice = mask[
        :,
        :,
        slice_index,
    ]

    image_slice = np.nan_to_num(
        image_slice,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ------------------------------------------------------------
    # Normalize for display
    # ------------------------------------------------------------

    min_value = float(
        image_slice.min()
    )

    max_value = float(
        image_slice.max()
    )

    if max_value > min_value:

        normalized = (
            image_slice
            - min_value
        ) / (
            max_value
            - min_value
        )

    else:

        normalized = np.zeros_like(
            image_slice,
            dtype=np.float32,
        )

    grayscale = (
        normalized
        * 255.0
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    # ------------------------------------------------------------
    # Base image
    # ------------------------------------------------------------

    base = Image.fromarray(
        grayscale,
        mode="L",
    ).convert(
        "RGBA"
    )

    rgba = np.array(
        base
    )

    tumor = (
        mask_slice > 0
    )

    # ------------------------------------------------------------
    # Tumor overlay
    # ------------------------------------------------------------

    rgba[
        tumor,
        0
    ] = 255

    rgba[
        tumor,
        1
    ] = 40

    rgba[
        tumor,
        2
    ] = 40

    rgba[
        tumor,
        3
    ] = 210

    overlay = Image.fromarray(
        rgba,
        mode="RGBA",
    )

    # ------------------------------------------------------------
    # PNG buffer
    # ------------------------------------------------------------

    buffer = io.BytesIO()

    overlay.save(
        buffer,
        format="PNG",
    )

    buffer.seek(0)

    return buffer.getvalue()


# ================================================================
# MASK RESIZING FOR VISUALIZATION
# ================================================================


def resize_mask_for_overlay(
    mask,
    target_shape=IMAGE_SHAPE,
):

    if mask.ndim != 3:

        raise ValueError(
            "Mask must be 3-dimensional."
        )

    if tuple(
        mask.shape
    ) == tuple(
        target_shape
    ):

        return (
            mask > 0
        ).astype(
            np.uint8
        )

    factors = [
        target_shape[i]
        / mask.shape[i]
        for i in range(3)
    ]

    resized = zoom(
        mask.astype(
            np.uint8
        ),
        factors,
        order=0,
    )

    if resized.shape != target_shape:

        resized = zoom(
            resized,
            [
                target_shape[i]
                / resized.shape[i]
                for i in range(3)
            ],
            order=0,
        )

    return (
        resized > 0
    ).astype(
        np.uint8
    )


# ================================================================
# MODEL INFORMATION HELPER
# ================================================================


def get_model_info():

    return {
        "model": MODEL_NAME,
        "architecture": ARCHITECTURE,
        "dataset": "BraTS 2021",
        "huggingface_repo": HF_REPO,
        "input_modalities": (
            INPUT_MODALITIES.copy()
        ),
        "input_shape": list(
            IMAGE_SHAPE
        ),
        "device": str(
            DEVICE
        ),
        "segmentation_threshold": (
            SEGMENTATION_THRESHOLD
        ),
        "minimum_tumor_fraction": (
            MIN_TUMOR_FRACTION
        ),
        "minimum_connected_voxels": (
            MIN_CONNECTED_VOXELS
        ),
    }
