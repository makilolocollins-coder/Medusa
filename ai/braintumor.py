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
import torch.nn.functional as F

from scipy.ndimage import zoom


# ================================================================
# CONFIGURATION
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SHAPE = (96, 96, 96)

HF_REPO = "Makky07/Brain_Tumor"

HF_MODEL_URL = (
    "https://huggingface.co/"
    "Makky07/Brain_Tumor/"
    "resolve/main/"
    "mammosense_brain_v31_best.pt"
)

MODEL_NAME = "MammoSense Brain V3.1"


# ================================================================
# 3D U-NET
# EXACT ARCHITECTURE USED DURING TRAINING
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

    def forward(self, x):

        return self.block(x)


class UNet3D(nn.Module):

    def __init__(self):

        super().__init__()

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

        self.bottleneck = DoubleConv3D(
            256,
            512,
        )

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

        self.output = nn.Conv3d(
            32,
            1,
            kernel_size=1,
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [
                d4,
                e4,
            ],
            dim=1,
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [
                d3,
                e3,
            ],
            dim=1,
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [
                d2,
                e2,
            ],
            dim=1,
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [
                d1,
                e1,
            ],
            dim=1,
        )

        d1 = self.dec1(d1)

        return self.output(d1)


# ================================================================
# MODEL DOWNLOAD
# ================================================================

@st.cache_resource(show_spinner=False)
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

    if not os.path.exists(model_path):

        urllib.request.urlretrieve(
            HF_MODEL_URL,
            model_path,
        )

    return model_path


# ================================================================
# LOAD MODEL
# ================================================================

@st.cache_resource(show_spinner=False)
def load_model():

    model_path = _download_model()

    model = UNet3D()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Invalid MammoSense Brain checkpoint."
        )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)

    model.eval()

    return model


# ================================================================
# NIFTI LOADING
# ================================================================

def load_nifti(file_bytes):

    if not file_bytes:

        raise ValueError(
            "Empty NIfTI file."
        )

    with tempfile.NamedTemporaryFile(
        suffix=".nii.gz",
        delete=False,
    ) as tmp:

        tmp.write(file_bytes)

        temp_path = tmp.name

    try:

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

    finally:

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

    return volume


# ================================================================
# MRI NORMALIZATION
# EXACT NORMALIZATION USED DURING TRAINING
# ================================================================

def normalize_mri(volume):

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
# RESIZE
# ================================================================

def resize_volume(
    volume,
    target_shape=IMAGE_SHAPE,
):

    factors = [
        target_shape[i]
        / volume.shape[i]
        for i in range(3)
    ]

    return zoom(
        volume,
        factors,
        order=1,
    ).astype(
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
        ("T1", t1_bytes),
        ("T1CE", t1ce_bytes),
        ("T2", t2_bytes),
        ("FLAIR", flair_bytes),
    ]

    channels = []

    original_shapes = {}

    for name, file_bytes in files:

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
        DEVICE
    )

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

    probabilities = torch.sigmoid(
        logits
    )

    probability_volume = (
        probabilities[0, 0]
        .float()
        .cpu()
        .numpy()
    )

    mask = (
        probability_volume >= 0.5
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
        / max(total_voxels, 1)
    )

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

    maximum_probability = float(
        probability_volume.max()
    )

    mean_probability = float(
        probability_volume.mean()
    )

    tumor_detected = (
        tumor_voxels > 0
    )

    if tumor_detected:

        prediction = (
            "TUMOR DETECTED"
        )

    else:

        prediction = (
            "NO TUMOR DETECTED"
        )

    return {
        "prediction": prediction,

        "confidence": confidence,

        "tumor_detected": tumor_detected,

        "tumor_voxels": tumor_voxels,

        "total_voxels": total_voxels,

        "tumor_fraction": tumor_fraction,

        "tumor_percentage": (
            tumor_fraction * 100.0
        ),

        "maximum_probability": (
            maximum_probability
        ),

        "mean_probability": (
            mean_probability
        ),

        "mask": mask,

        "probability_volume": (
            probability_volume
        ),

        "original_shapes": (
            original_shapes
        ),

        "device": str(
            DEVICE
        ),

        "model": MODEL_NAME,

        "architecture": "3D_U-Net",

        "input_shape": list(
            IMAGE_SHAPE
        ),

        "input_modalities": [
            "T1",
            "T1CE",
            "T2",
            "FLAIR",
        ],
    }


# ================================================================
# REPRESENTATIVE MRI SLICE
# ================================================================

def create_overlay(
    flair_bytes,
    mask,
):

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

    # Middle axial slice
    slice_index = (
        flair.shape[2] // 2
    )

    image_slice = flair[
        :, :,
        slice_index
    ]

    mask_slice = mask[
        :, :,
        slice_index
    ]

    image_slice = np.nan_to_num(
        image_slice
    )

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
            image_slice
        )

    grayscale = (
        normalized * 255
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

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

    buffer = io.BytesIO()

    overlay.save(
        buffer,
        format="PNG",
    )

    buffer.seek(0)

    return buffer.getvalue()
