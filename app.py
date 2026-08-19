# ================================================================
# MEDUSA AI V2
# Premium AI Health Platform
#
# Frontend + MammoSense AI
#
# Hugging Face:
# Makky07/MammoSense-breast-ultrasound
#
# Model:
# mammosense_v2.pt
#
# ================================================================

import json
import hashlib
from datetime import datetime

import streamlit as st
import torch
import timm

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Medusa AI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ================================================================
# MEDUSA DESIGN SYSTEM
# ================================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F8FA;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}


/* ============================================================
   HIDE STREAMLIT DEFAULT UI
   ============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ============================================================
   TOP NAV
   ============================================================ */

.medusa-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0 35px 0;
}

.medusa-brand {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #101828;
}

.medusa-brand span {
    color: #6D5DFB;
}

.nav-status {
    background: #ECFDF3;
    color: #027A48;
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    padding: 25px 0 30px 0;
}

.hero h1 {
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1.05;
    letter-spacing: -2.5px;
    color: #101828;
    margin-bottom: 15px;
}

.hero h1 span {
    color: #6D5DFB;
}

.hero p {
    color: #667085;
    font-size: 17px;
    max-width: 620px;
    line-height: 1.7;
}


/* ============================================================
   PREMIUM CARD
   ============================================================ */

.card {
    background: #FFFFFF;
    border: 1px solid #EAECF0;
    border-radius: 24px;
    padding: 27px;
    box-shadow:
        0 8px 30px rgba(16, 24, 40, 0.035);
}

.card-title {
    color: #101828;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-subtitle {
    color: #667085;
    font-size: 14px;
    line-height: 1.6;
}


/* ============================================================
   AI ACTION CARD
   ============================================================ */

.ai-card {
    background:
        linear-gradient(
            135deg,
            #15112D 0%,
            #29205A 100%
        );

    border-radius: 28px;
    padding: 35px;
    color: white;
    min-height: 280px;
    position: relative;
    overflow: hidden;
}

.ai-card:after {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    border-radius: 50%;
    background: rgba(139, 92, 246, 0.20);
    right: -80px;
    top: -80px;
}

.ai-small {
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity: 0.65;
}

.ai-title {
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
    letter-spacing: -1px;
}

.ai-description {
    max-width: 430px;
    font-size: 14px;
    line-height: 1.7;
    opacity: 0.75;
}


/* ============================================================
   MODEL CARDS
   ============================================================ */

.model-card {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 20px;
    padding: 22px;
    min-height: 170px;
}

.model-icon {
    font-size: 27px;
    margin-bottom: 13px;
}

.model-name {
    font-weight: 700;
    color: #101828;
}

.model-info {
    color: #667085;
    font-size: 13px;
    margin-top: 7px;
    line-height: 1.5;
}

.available {
    display: inline-block;
    margin-top: 15px;
    color: #027A48;
    background: #ECFDF3;
    padding: 5px 9px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}

.coming {
    display: inline-block;
    margin-top: 15px;
    color: #667085;
    background: #F2F4F7;
    padding: 5px 9px;
    border-radius: 20px;
    font-size: 11px;
}


/* ============================================================
   RESULT
   ============================================================ */

.result-main {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 28px;
    padding: 35px;
    text-align: center;
}

.result-caption {
    color: #667085;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 1.5px;
}

.result-value {
    font-size: 44px;
    font-weight: 800;
    color: #101828;
    margin-top: 8px;
}

.result-confidence {
    color: #667085;
    font-size: 16px;
}


/* ============================================================
   PROBABILITY
   ============================================================ */

.probability-box {
    background: #F9FAFB;
    border-radius: 16px;
    padding: 18px;
    margin-top: 10px;
}


/* ============================================================
   HEALTH TIMELINE
   ============================================================ */

.timeline-item {
    border-left: 2px solid #E4E7EC;
    padding-left: 20px;
    padding-bottom: 25px;
    margin-left: 7px;
}

.timeline-dot {
    width: 10px;
    height: 10px;
    background: #6D5DFB;
    border-radius: 50%;
    position: absolute;
    margin-left: -26px;
    margin-top: 5px;
}

.timeline-title {
    font-weight: 700;
    color: #101828;
}

.timeline-info {
    color: #667085;
    font-size: 13px;
    margin-top: 4px;
}


/* ============================================================
   MARKETPLACE
   ============================================================ */

.market-card {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 22px;
    padding: 25px;
    min-height: 190px;
    transition: 0.2s;
}

.market-icon {
    font-size: 30px;
}

.market-title {
    font-size: 17px;
    font-weight: 700;
    color: #101828;
    margin-top: 15px;
}

.market-text {
    color: #667085;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 7px;
}


/* ============================================================
   DISCLAIMER
   ============================================================ */

.disclaimer {
    background: #FFF8E7;
    border: 1px solid #F2CC72;
    color: #694B00;
    padding: 16px 19px;
    border-radius: 15px;
    font-size: 12px;
    line-height: 1.6;
}


/* ============================================================
   SECTION
   ============================================================ */

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #101828;
    letter-spacing: -0.8px;
    margin-top: 35px;
    margin-bottom: 18px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.medusa-footer {
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
# MODEL SETTINGS
# ================================================================

HF_REPO = (
    "Makky07/MammoSense-breast-ultrasound"
)

MODEL_FILE = "mammosense_v2.pt"

CONFIG_FILE = "mammosense_v2_config.json"

IMAGE_SIZE = 224


# ================================================================
# LOAD MAMMOSENSE
# ================================================================

@st.cache_resource(show_spinner=False)
def load_mammosense():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------

    config_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=CONFIG_FILE,
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)


    classes = config.get(
        "class_names",
        [
            "Normal",
            "Benign",
            "Malignant",
        ],
    )


    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILE,
    )


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
    # STATE DICT
    # ------------------------------------------------------------

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


    cleaned_state = {}

    for key, value in state_dict.items():

        key = key.replace(
            "module.",
            "",
        )

        key = key.replace(
            "model.",
            "",
        )

        cleaned_state[key] = value


    missing, unexpected = (
        model.load_state_dict(
            cleaned_state,
            strict=False,
        )
    )


    if missing:

        raise RuntimeError(
            f"Model loading failed. "
            f"Missing keys: {len(missing)}"
        )


    model = model.to(device)

    model.eval()


    # ------------------------------------------------------------
    # TRANSFORM
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
                [
                    0.485,
                    0.456,
                    0.406,
                ],
                [
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
    )


# ================================================================
# PREDICT
# ================================================================

@torch.inference_mode()
def run_prediction(
    image,
    model,
    transform,
    classes,
    device,
):

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    output = model(tensor)

    probabilities = torch.softmax(
        output,
        dim=1,
    )[0]


    index = int(
        torch.argmax(
            probabilities
        ).item()
    )


    result = {
        name: float(
            probabilities[i].item()
        )

        for i, name in enumerate(
            classes
        )
    }


    return {
        "prediction": classes[index],
        "confidence": float(
            probabilities[index].item()
        ),
        "probabilities": result,
    }


# ================================================================
# TOP NAVIGATION
# ================================================================

st.markdown(
    """
<div class="medusa-nav">

    <div class="medusa-brand">
        MEDUSA<span>◉</span>
    </div>

    <div class="nav-status">
        ● AI SYSTEM ONLINE
    </div>

</div>
""",
    unsafe_allow_html=True,
)


# ================================================================
# NAV BUTTONS
# ================================================================

nav1, nav2, nav3, nav4, nav5 = st.columns(
    [1, 1, 1, 1, 1]
)


with nav1:

    if st.button(
        "Home",
        use_container_width=True,
    ):
        st.session_state.page = "Home"
        st.rerun()


with nav2:

    if st.button(
        "AI Detection",
        use_container_width=True,
    ):
        st.session_state.page = "Detection"
        st.rerun()


with nav3:

    if st.button(
        "Health",
        use_container_width=True,
    ):
        st.session_state.page = "Health"
        st.rerun()


with nav4:

    if st.button(
        "Marketplace",
        use_container_width=True,
    ):
        st.session_state.page = "Marketplace"
        st.rerun()


with nav5:

    if st.button(
        "Profile",
        use_container_width=True,
    ):
        st.session_state.page = "Profile"
        st.rerun()


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
                Medusa brings AI-powered health analysis,
                personal health insights and healthcare
                services into one intelligent platform.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ------------------------------------------------------------
    # MAIN AI CARD
    # ------------------------------------------------------------

    st.markdown(
        """
        <div class="ai-card">

            <div class="ai-small">
                MEDUSA INTELLIGENCE
            </div>

            <div class="ai-title">
                AI Health Detection
            </div>

            <div class="ai-description">
                Analyse medical images using our
                AI-powered screening models.
                Start with MammoSense breast
                ultrasound analysis.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown("")


    if st.button(
        "Start an AI Analysis →",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.page = "Detection"
        st.rerun()


    # ------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Medusa Intelligence'
        '</div>',
        unsafe_allow_html=True,
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown(
            """
            <div class="model-card">

                <div class="model-icon">
                    🧬
                </div>

                <div class="model-name">
                    MammoSense
                </div>

                <div class="model-info">
                    Breast ultrasound AI
                    classification.
                </div>

                <span class="available">
                    AVAILABLE
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with c2:

        st.markdown(
            """
            <div class="model-card">

                <div class="model-icon">
                    🧠
                </div>

                <div class="model-name">
                    Prostate AI
                </div>

                <div class="model-info">
                    Multimodal prostate MRI
                    intelligence.
                </div>

                <span class="coming">
                    COMING SOON
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with c3:

        st.markdown(
            """
            <div class="model-card">

                <div class="model-icon">
                    ✦
                </div>

                <div class="model-name">
                    More Intelligence
                </div>

                <div class="model-info">
                    Additional health AI
                    models will appear here.
                </div>

                <span class="coming">
                    EXPANDING
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ------------------------------------------------------------
    # RECENT ANALYSIS
    # ------------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Recent activity'
        '</div>',
        unsafe_allow_html=True,
    )


    if not st.session_state.history:

        st.info(
            "Your recent AI analyses will appear here."
        )

    else:

        for item in reversed(
            st.session_state.history[-3:]
        ):

            st.markdown(
                f"""
                <div class="card">

                    <div class="card-title">
                        {item["prediction"]}
                    </div>

                    <div class="card-subtitle">
                        MammoSense ·
                        {item["confidence"] * 100:.1f}%
                        confidence ·
                        {item["time"]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ================================================================
# AI DETECTION
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
                for MammoSense analysis.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------

    try:

        with st.spinner(
            "Preparing MammoSense..."
        ):

            (
                model,
                transform,
                classes,
                device,
            ) = load_mammosense()

        st.success(
            f"MammoSense ready · {device}"
        )

    except Exception as error:

        st.error(
            "MammoSense could not be loaded."
        )

        st.code(
            str(error)
        )

        st.stop()


    # ------------------------------------------------------------
    # UPLOAD
    # ------------------------------------------------------------

    uploaded = st.file_uploader(
        "Upload ultrasound image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )


    if uploaded:

        image = Image.open(
            uploaded
        )


        left, right = st.columns(
            [1.15, 1]
        )


        with left:

            st.image(
                image,
                caption="Ultrasound image",
                use_container_width=True,
            )


        with right:

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        Ready for analysis
                    </div>

                    <div class="card-subtitle">
                        Your image will be processed
                        by the MammoSense AI model.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


            st.write(
                f"**Image:** {uploaded.name}"
            )

            st.write(
                f"**Size:** "
                f"{image.width} × "
                f"{image.height}"
            )


            analyse = st.button(
                "Analyse with MammoSense →",
                type="primary",
                use_container_width=True,
            )


            if analyse:

                with st.spinner(
                    "Medusa is analysing the image..."
                ):

                    result = run_prediction(
                        image,
                        model,
                        transform,
                        classes,
                        device,
                    )


                st.session_state.prediction = result


                image_hash = hashlib.md5(
                    uploaded.getvalue()
                ).hexdigest()


                st.session_state.history.append(
                    {
                        "prediction":
                            result["prediction"],

                        "confidence":
                            result["confidence"],

                        "time":
                            datetime.now().strftime(
                                "%d %b %Y · %H:%M"
                            ),

                        "hash":
                            image_hash,
                    }
                )


                st.rerun()


    # ------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------

    if st.session_state.prediction:

        result = (
            st.session_state.prediction
        )


        st.markdown("---")


        st.markdown(
            """
            <div class="section-title">
                Analysis result
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.markdown(
            f"""
            <div class="result-main">

                <div class="result-caption">
                    MammoSense AI classification
                </div>

                <div class="result-value">
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

            st.markdown(
                f"**{name}**"
            )

            st.progress(
                probability
            )

            st.caption(
                f"{probability * 100:.2f}%"
            )


        # --------------------------------------------------------
        # AI INSIGHT
        # --------------------------------------------------------

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    AI Insight
                </div>

                <div class="card-subtitle">
                    MammoSense classified this image
                    using its trained breast ultrasound
                    classification model.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # --------------------------------------------------------
        # NEXT STEPS
        # --------------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'What next?'
            '</div>',
            unsafe_allow_html=True,
        )


        a, b, c = st.columns(3)


        with a:

            st.button(
                "👨‍⚕️ Review with a doctor",
                use_container_width=True,
            )


        with b:

            st.button(
                "🏥 Find diagnostic centre",
                use_container_width=True,
            )


        with c:

            st.button(
                "📄 Save report",
                use_container_width=True,
            )


        st.markdown("")


        st.markdown(
            """
            <div class="disclaimer">

            ⚠️ <b>Medical notice:</b>
            Medusa is an AI-assisted screening
            platform. This result is not a medical
            diagnosis and should be reviewed by a
            qualified healthcare professional.

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
                A simple timeline of your
                Medusa activity.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    if not st.session_state.history:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Your health timeline is empty.
                </div>

                <div class="card-subtitle">
                    Complete your first AI analysis
                    to begin building your timeline.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        for item in reversed(
            st.session_state.history
        ):

            st.markdown(
                f"""
                <div class="timeline-item">

                    <div class="timeline-dot">
                    </div>

                    <div class="timeline-title">
                        MammoSense Analysis
                    </div>

                    <div class="timeline-info">
                        {item["prediction"]}
                        ·
                        {item["confidence"] * 100:.1f}%
                        confidence
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
                Healthcare,
                <span>connected.</span>
            </h1>

            <p>
                Discover healthcare professionals,
                diagnostic services and health products
                through the Medusa ecosystem.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown(
            """
            <div class="market-card">

                <div class="market-icon">
                    👨‍⚕️
                </div>

                <div class="market-title">
                    Doctors
                </div>

                <div class="market-text">
                    Find qualified healthcare
                    professionals and specialists.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with c2:

        st.markdown(
            """
            <div class="market-card">

                <div class="market-icon">
                    🏥
                </div>

                <div class="market-title">
                    Diagnostics
                </div>

                <div class="market-text">
                    Find imaging centres,
                    laboratories and diagnostic
                    services.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with c3:

        st.markdown(
            """
            <div class="market-card">

                <div class="market-icon">
                    💊
                </div>

                <div class="market-title">
                    Pharmacy
                </div>

                <div class="market-text">
                    Access trusted healthcare
                    products and pharmacy services.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown("")


    st.info(
        "Marketplace providers will be connected "
        "in a future Medusa release."
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
                Manage your Medusa account and
                preferences.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Medusa User
            </div>

            <div class="card-subtitle">
                Account management, privacy,
                notifications and health preferences
                will appear here.
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
    <div class="medusa-footer">

        MEDUSA AI · Intelligent health infrastructure

        <br><br>

        AI-assisted screening only.
        Not a substitute for professional medical advice.

    </div>
    """,
    unsafe_allow_html=True,
)
