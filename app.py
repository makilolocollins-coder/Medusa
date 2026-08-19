# ================================================================
# MEDUSA AI
# Streamlit Cloud Deployment Version
#
# AI Health Intelligence Platform
#
# Current model:
# MammoSense - Breast Ultrasound
#
# Hugging Face:
# https://huggingface.co/Makky07/MammoSense-breast-ultrasound
#
# ================================================================

import os
import json
import hashlib
from datetime import datetime

import streamlit as st
import torch
import timm

from PIL import Image
from torchvision import transforms
from huggingface_hub import (
    HfApi,
    hf_hub_download,
)


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Medusa AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ================================================================
# DESIGN
# ================================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F8FA;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}


/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* Brand */

.brand {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1.2px;
    color: #101828;
}

.brand span {
    color: #6D5DFB;
}


/* Status */

.status {
    display: inline-block;
    background: #ECFDF3;
    color: #027A48;
    padding: 7px 12px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
}


/* Hero */

.hero {
    padding: 45px 0 30px 0;
}

.hero h1 {
    font-size: clamp(36px, 6vw, 62px);
    line-height: 1.03;
    letter-spacing: -3px;
    color: #101828;
}

.hero h1 span {
    color: #6D5DFB;
}

.hero p {
    max-width: 650px;
    color: #667085;
    font-size: 17px;
    line-height: 1.7;
}


/* Cards */

.card {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 24px;
    padding: 26px;
    box-shadow:
        0 8px 30px rgba(16,24,40,0.035);
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #101828;
}

.card-text {
    color: #667085;
    font-size: 13px;
    line-height: 1.6;
}


/* AI card */

.ai-card {
    background:
        linear-gradient(
            135deg,
            #12101F,
            #30265E
        );

    border-radius: 28px;
    padding: 38px;
    color: white;
    min-height: 260px;
}

.ai-label {
    font-size: 11px;
    letter-spacing: 1.5px;
    opacity: .65;
}

.ai-title {
    font-size: 32px;
    font-weight: 800;
    margin-top: 12px;
}

.ai-text {
    max-width: 550px;
    color: rgba(255,255,255,.72);
    line-height: 1.7;
    font-size: 14px;
}


/* Model */

.model-card {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 20px;
    padding: 22px;
    min-height: 160px;
}

.model-icon {
    font-size: 28px;
}

.model-name {
    font-weight: 700;
    margin-top: 12px;
    color: #101828;
}

.model-description {
    color: #667085;
    font-size: 13px;
    margin-top: 7px;
}


/* Result */

.result {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 28px;
    padding: 35px;
    text-align: center;
}

.result-small {
    color: #667085;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.result-name {
    color: #101828;
    font-size: 44px;
    font-weight: 800;
    margin-top: 8px;
}

.result-confidence {
    color: #667085;
}


/* Warning */

.warning {
    background: #FFF8E7;
    border: 1px solid #F2CC72;
    color: #694B00;
    border-radius: 15px;
    padding: 15px;
    font-size: 12px;
    line-height: 1.6;
}


/* Section */

.section {
    font-size: 25px;
    font-weight: 800;
    color: #101828;
    letter-spacing: -.8px;
    margin-top: 38px;
    margin-bottom: 18px;
}


/* Marketplace */

.market {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 22px;
    padding: 25px;
    min-height: 180px;
}

.market-icon {
    font-size: 30px;
}

.market-title {
    font-size: 17px;
    font-weight: 700;
    margin-top: 12px;
}

.market-text {
    color: #667085;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 7px;
}


/* Footer */

.footer {
    text-align: center;
    color: #98A2B3;
    font-size: 11px;
    margin-top: 70px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ================================================================
# SESSION STATE
# ================================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ================================================================
# HUGGING FACE CONFIG
# ================================================================

HF_REPO = (
    "Makky07/MammoSense-breast-ultrasound"
)

IMAGE_SIZE = 224


# ================================================================
# FIND MODEL FILE
# ================================================================

@st.cache_data(ttl=3600)
def find_model_files():

    api = HfApi()

    files = api.list_repo_files(
        repo_id=HF_REPO,
        repo_type="model",
    )

    model_candidates = [
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

    config_candidates = [
        file
        for file in files
        if file.lower().endswith(
            ".json"
        )
    ]

    if not model_candidates:

        raise RuntimeError(
            "No PyTorch model file was found "
            "in the Hugging Face repository."
        )


    # Prefer MammoSense / GAIA model names

    preferred = [
        file
        for file in model_candidates
        if (
            "mammosense" in file.lower()
            or "gaia" in file.lower()
        )
    ]


    if preferred:

        model_file = preferred[0]

    else:

        model_file = model_candidates[0]


    config_file = None

    preferred_configs = [
        file
        for file in config_candidates
        if (
            "config" in file.lower()
            or "metadata" in file.lower()
        )
    ]


    if preferred_configs:

        config_file = preferred_configs[0]


    return model_file, config_file


# ================================================================
# LOAD MODEL
# ================================================================

@st.cache_resource(
    show_spinner=False,
)
def load_mammosense():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # ------------------------------------------------------------
    # FIND FILES
    # ------------------------------------------------------------

    model_file, config_file = (
        find_model_files()
    )


    # ------------------------------------------------------------
    # DOWNLOAD MODEL
    # ------------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=model_file,
        repo_type="model",
    )


    # ------------------------------------------------------------
    # CLASSES
    # ------------------------------------------------------------

    classes = [
        "Normal",
        "Benign",
        "Malignant",
    ]


    # Try config if available

    if config_file:

        try:

            config_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=config_file,
                repo_type="model",
            )


            with open(
                config_path,
                "r",
                encoding="utf-8",
            ) as f:

                config = json.load(f)


            config_classes = config.get(
                "class_names"
            )


            if (
                isinstance(
                    config_classes,
                    list,
                )
                and len(config_classes) > 0
            ):

                classes = config_classes


        except Exception:

            pass


    # ------------------------------------------------------------
    # CREATE ARCHITECTURE
    # ------------------------------------------------------------

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=len(classes),
    )


    # ------------------------------------------------------------
    # PYTORCH 2.6
    # ------------------------------------------------------------

    try:

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )

    except Exception:

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )


    # ------------------------------------------------------------
    # EXTRACT STATE DICT
    # ------------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        raise RuntimeError(
            "The checkpoint format is not "
            "supported."
        )


    # ------------------------------------------------------------
    # CLEAN STATE DICT
    # ------------------------------------------------------------

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

        cleaned[new_key] = value


    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    missing, unexpected = (
        model.load_state_dict(
            cleaned,
            strict=False,
        )
    )


    # Don't silently run a broken model

    if missing:

        raise RuntimeError(
            "MammoSense checkpoint does not "
            "match the expected ViT architecture."
        )


    # ------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------

    model = model.to(device)

    model.eval()


    # ------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------

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


    return (
        model,
        transform,
        classes,
        device,
        model_file,
    )


# ================================================================
# PREDICTION
# ================================================================

@torch.inference_mode()
def predict(
    image,
    model,
    transform,
    classes,
    device,
):

    image = image.convert("RGB")

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

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


    results = {}

    for i, name in enumerate(
        classes
    ):

        results[name] = float(
            probabilities[i].item()
        )


    return {
        "prediction": classes[index],

        "confidence": float(
            probabilities[index].item()
        ),

        "probabilities": results,
    }


# ================================================================
# TOP BRAND
# ================================================================

left, right = st.columns(
    [4, 1]
)

with left:

    st.markdown(
        """
        <div class="brand">
            MEDUSA<span>◉</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:

    st.markdown(
        """
        <div style="text-align:right;">
            <span class="status">
                ● AI ONLINE
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("")


# ================================================================
# NAVIGATION
# ================================================================

n1, n2, n3, n4, n5 = st.columns(5)


def navigate(page):

    st.session_state.page = page
    st.rerun()


with n1:

    if st.button(
        "Home",
        use_container_width=True,
    ):

        navigate("Home")


with n2:

    if st.button(
        "AI Detection",
        use_container_width=True,
    ):

        navigate("Detection")


with n3:

    if st.button(
        "Health",
        use_container_width=True,
    ):

        navigate("Health")


with n4:

    if st.button(
        "Marketplace",
        use_container_width=True,
    ):

        navigate("Marketplace")


with n5:

    if st.button(
        "Profile",
        use_container_width=True,
    ):

        navigate("Profile")


st.markdown("---")


# ================================================================
# HOME
# ================================================================

if st.session_state.page == "Home":

    st.markdown(
        """
        <div class="hero">

            <h1>
                Your health,<br>
                <span>intelligently connected.</span>
            </h1>

            <p>
                Medusa brings AI-powered health
                analysis, personal health insights
                and healthcare services together
                in one intelligent platform.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="ai-card">

            <div class="ai-label">
                MEDUSA INTELLIGENCE
            </div>

            <div class="ai-title">
                AI Health Detection
            </div>

            <div class="ai-text">
                Start with MammoSense,
                Medusa's breast ultrasound
                intelligence engine.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown("")


    if st.button(
        "Start AI Analysis →",
        type="primary",
        use_container_width=True,
    ):

        navigate("Detection")


    st.markdown(
        '<div class="section">'
        'AI Models'
        '</div>',
        unsafe_allow_html=True,
    )


    a, b, c = st.columns(3)


    with a:

        st.markdown(
            """
            <div class="model-card">

                <div class="model-icon">
                    🧬
                </div>

                <div class="model-name">
                    MammoSense
                </div>

                <div class="model-description">
                    Breast ultrasound
                    classification.
                </div>

                <span class="available">
                    AVAILABLE
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with b:

        st.markdown(
            """
            <div class="model-card">

                <div class="model-icon">
                    🧠
                </div>

                <div class="model-name">
                    Prostate AI
                </div>

                <div class="model-description">
                    Multimodal prostate
                    MRI intelligence.
                </div>

                <span class="coming">
                    COMING SOON
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with c:

        st.markdown(
            """
            <div class="model-card">

                <div class="model-icon">
                    ✦
                </div>

                <div class="model-name">
                    Medusa Intelligence
                </div>

                <div class="model-description">
                    More AI health models
                    will be added.
                </div>

                <span class="coming">
                    EXPANDING
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ================================================================
# DETECTION
# ================================================================

elif st.session_state.page == "Detection":

    st.markdown(
        """
        <div class="hero">

            <h1>
                AI Detection
            </h1>

            <p>
                Upload a breast ultrasound image
                and let MammoSense analyse it.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ------------------------------------------------------------
    # MODEL STATUS
    # ------------------------------------------------------------

    with st.spinner(
        "Preparing MammoSense..."
    ):

        try:

            (
                model,
                transform,
                classes,
                device,
                model_file,
            ) = load_mammosense()

            model_ready = True

        except Exception as error:

            model_ready = False

            st.error(
                "MammoSense could not be loaded."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    str(error)
                )


    if model_ready:

        st.success(
            f"MammoSense ready · "
            f"{device.type.upper()}"
        )


        uploaded = st.file_uploader(
            "Upload breast ultrasound",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
        )


        if uploaded:

            try:

                image = Image.open(
                    uploaded
                )

                image.load()

            except Exception:

                st.error(
                    "This image could not be read."
                )

                st.stop()


            left, right = st.columns(
                [1.15, 1]
            )


            with left:

                st.image(
                    image,
                    caption="Uploaded image",
                    use_container_width=True,
                )


            with right:

                st.markdown(
                    """
                    <div class="card">

                        <div class="card-title">
                            Image ready
                        </div>

                        <br>

                        <div class="card-text">
                            The image will be
                            processed by the
                            MammoSense model.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                st.write(
                    f"**File:** {uploaded.name}"
                )

                st.write(
                    f"**Dimensions:** "
                    f"{image.width} × "
                    f"{image.height}"
                )


                if st.button(
                    "Analyse with MammoSense →",
                    type="primary",
                    use_container_width=True,
                ):

                    with st.spinner(
                        "Medusa is analysing..."
                    ):

                        result = predict(
                            image,
                            model,
                            transform,
                            classes,
                            device,
                        )


                    st.session_state.prediction = (
                        result
                    )


                    file_hash = hashlib.md5(
                        uploaded.getvalue()
                    ).hexdigest()


                    st.session_state.history.append(
                        {
                            "prediction":
                                result[
                                    "prediction"
                                ],

                            "confidence":
                                result[
                                    "confidence"
                                ],

                            "time":
                                datetime.now().strftime(
                                    "%d %b %Y · %H:%M"
                                ),

                            "hash":
                                file_hash,
                        }
                    )


                    st.rerun()


    # ------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------

    if (
        st.session_state.prediction
        and model_ready
    ):

        result = (
            st.session_state.prediction
        )


        st.markdown("---")


        st.markdown(
            '<div class="section">'
            'Analysis Result'
            '</div>',
            unsafe_allow_html=True,
        )


        st.markdown(
            f"""
            <div class="result">

                <div class="result-small">
                    MammoSense classification
                </div>

                <div class="result-name">
                    {result["prediction"]}
                </div>

                <div class="result-confidence">
                    {result["confidence"] * 100:.2f}%
                    confidence
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        st.markdown(
            "### Probability distribution"
        )


        for (
            name,
            probability,
        ) in result[
            "probabilities"
        ].items():

            st.write(
                f"**{name}**"
            )

            st.progress(
                probability
            )

            st.caption(
                f"{probability * 100:.2f}%"
            )


        st.markdown(
            '<div class="section">'
            'What next?'
            '</div>',
            unsafe_allow_html=True,
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.button(
                "👨‍⚕️ Review with doctor",
                use_container_width=True,
            )


        with c2:

            st.button(
                "🏥 Find diagnostic centre",
                use_container_width=True,
            )


        with c3:

            st.button(
                "📄 Save report",
                use_container_width=True,
            )


        st.markdown("")


        st.markdown(
            """
            <div class="warning">

            ⚠️ <b>Medical notice:</b>
            Medusa provides AI-assisted screening
            support. This result is not a diagnosis
            and should be reviewed by a qualified
            healthcare professional.

            </div>
            """,
            unsafe_allow_html=True,
        )


# ================================================================
# HEALTH
# ================================================================

elif st.session_state.page == "Health":

    st.markdown(
        """
        <div class="hero">

            <h1>
                Your Health
            </h1>

            <p>
                Your Medusa activity and AI
                analysis history.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    if not st.session_state.history:

        st.info(
            "Your health timeline is empty."
        )

    else:

        for item in reversed(
            st.session_state.history
        ):

            st.markdown(
                f"""
                <div class="card">

                    <div class="card-title">
                        🧬 MammoSense Analysis
                    </div>

                    <div class="card-text">
                        Result:
                        <b>
                        {item["prediction"]}
                        </b>
                        <br>
                        Confidence:
                        {item["confidence"] * 100:.1f}%
                        <br>
                        {item["time"]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ================================================================
# MARKETPLACE
# ================================================================

elif st.session_state.page == "Marketplace":

    st.markdown(
        """
        <div class="hero">

            <h1>
                Healthcare,<br>
                <span>connected.</span>
            </h1>

            <p>
                Discover doctors, diagnostic centres,
                pharmacies and healthcare services.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    a, b, c = st.columns(3)


    with a:

        st.markdown(
            """
            <div class="market">

                <div class="market-icon">
                    👨‍⚕️
                </div>

                <div class="market-title">
                    Doctors
                </div>

                <div class="market-text">
                    Connect with qualified
                    healthcare professionals.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with b:

        st.markdown(
            """
            <div class="market">

                <div class="market-icon">
                    🏥
                </div>

                <div class="market-title">
                    Diagnostics
                </div>

                <div class="market-text">
                    Discover imaging centres,
                    laboratories and diagnostic
                    services.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with c:

        st.markdown(
            """
            <div class="market">

                <div class="market-icon">
                    💊
                </div>

                <div class="market-title">
                    Pharmacy
                </div>

                <div class="market-text">
                    Access trusted healthcare
                    products and services.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown("")

    st.info(
        "Marketplace services will be connected "
        "as Medusa expands."
    )


# ================================================================
# PROFILE
# ================================================================

elif st.session_state.page == "Profile":

    st.markdown(
        """
        <div class="hero">

            <h1>
                Profile
            </h1>

            <p>
                Manage your Medusa experience.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                👤 Medusa User
            </div>

            <br>

            <div class="card-text">
                Account, privacy, notification
                and health preferences will
                appear here.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# FOOTER
# ================================================================

st.markdown(
    """
    <div class="footer">

        MEDUSA AI · Intelligent Health Infrastructure

        <br><br>

        AI-assisted screening only.
        Not a substitute for professional medical advice.

    </div>
    """,
    unsafe_allow_html=True,
)
