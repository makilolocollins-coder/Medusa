import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIG
# ============================================================

HF_REPO = "Makky07/Lung_tb"

MODEL_FILE = "medusa_lung_cancer_tb_fixed.pt"

IMAGE_SIZE = 384

CLASSES = [
    "Lung Cancer",
    "Tuberculosis",
]


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim=128,
        num_heads=4,
        mlp_ratio=4,
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(
                dim,
                dim * mlp_ratio,
            ),
            nn.GELU(),
            nn.Linear(
                dim * mlp_ratio,
                dim,
            ),
        )

    def forward(self, x):

        y = self.norm1(x)

        y = self.attn(
            y,
            y,
            y,
        )[0]

        x = x + y

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ============================================================
# TINY VIT
# ============================================================

class TinyViT(nn.Module):

    def __init__(
        self,
        img_size=384,
        patch_size=16,
        in_chans=1,
        embed_dim=128,
        depth=4,
        num_heads=4,
    ):

        super().__init__()

        self.img_size = img_size

        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (
            img_size // patch_size
        ) ** 2

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                embed_dim,
            )
        )

        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                num_patches + 1,
                embed_dim,
            )
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_heads,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

    def forward(self, x):

        x = self.patch_embed(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        batch_size = x.shape[0]

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            [cls, x],
            dim=1,
        )

        x = x + self.pos_embed

        x = self.blocks(x)

        return self.norm(x)


# ============================================================
# CLASSIFIER
# ============================================================

class MultiLabelClassifier(nn.Module):

    def __init__(
        self,
        num_classes=2,
        img_size=384,
    ):

        super().__init__()

        self.encoder = TinyViT(
            img_size=img_size
        )

        self.head = nn.Linear(
            128,
            num_classes,
        )

    def forward(self, x):

        features = self.encoder(x)

        return self.head(
            features[:, 0, :]
        )


# ============================================================
# MODEL
# ============================================================

_MODEL = None


def load_model():

    global _MODEL

    if _MODEL is not None:

        return _MODEL

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILE,
        repo_type="model",
    )

    model = MultiLabelClassifier(
        num_classes=2,
        img_size=384,
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint,
        strict=False,
    )

    model.eval()

    _MODEL = model

    return model


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(image):

    image = image.convert("L")

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    array = array / 255.0

    tensor = torch.from_numpy(
        array
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# PREDICTION
# ============================================================

@torch.inference_mode()
def predict(image):

    model = load_model()

    tensor = preprocess(image)

    logits = model(tensor)

    probabilities = torch.sigmoid(
        logits
    )[0]

    lung_cancer = float(
        probabilities[0].item()
    )

    tuberculosis = float(
        probabilities[1].item()
    )

    findings = []

    if lung_cancer >= 0.5:

        findings.append(
            "Lung Cancer"
        )

    if tuberculosis >= 0.5:

        findings.append(
            "Tuberculosis"
        )

    if not findings:

        findings.append(
            "No significant finding"
        )

    return {

        "findings": findings,

        "lung_cancer":
            lung_cancer,

        "tuberculosis":
            tuberculosis,

        "probabilities": {

            "Lung Cancer":
                lung_cancer,

            "Tuberculosis":
                tuberculosis,
        },

        "model":
            "Medusa Lung AI",

        "architecture":
            "TinyViT",

        "image_size":
            IMAGE_SIZE,
    }
