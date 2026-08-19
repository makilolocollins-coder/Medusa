# ================================================================
# MEDUSA AI
# Deployment-ready Streamlit application
#
# Current AI:
# MammoSense - Breast Ultrasound
#
# Hugging Face:
# Makky07/MammoSense-breast-ultrasound
# ================================================================

import json
from datetime import datetime

import streamlit as st
import torch
import timm

from PIL import Image
from torchvision import transforms
from huggingface_hub import HfApi, hf_hub_download


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
# CONFIGURATION
# ================================================================

HF_REPO = "Makky07/MammoSense-breast-ultrasound"

IMAGE_SIZE = 224


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #F7F8FA;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* BRAND */

.brand {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1.3px;
    color: #101828;
}

.brand span {
    color: #6D5DFB;
}


/* STATUS */

.status {
    display: inline-block;
    background: #ECFDF3;
    color: #027A48;
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
}


/* HERO */

.hero {
    padding: 45px 0 30px 0;
}

.hero h1 {
    font-size: clamp(38px, 6vw, 64px);
    line-height: 1.02;
    letter-spacing: -3px;
    color: #101828;
}

.hero h1 span {
    color: #6D5DFB;
}

.hero p {
    max-width: 680px;
    color: #667085;
    font-size: 17px;
    line-height: 1.7;
}


/* AI CARD */

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
    min-height: 250px;
}

.ai-label {
    font-size: 11px;
    letter-spacing: 1.7px;
    opacity: .65;
}

.ai-title {
    font-size: 34px;
    font-weight: 800;
    margin-top: 13px;
}

.ai-text {
    max-width: 580px;
    color: rgba(255,255,255,.72);
    line-height: 1.7;
    font-size: 14px;
}


/* CARDS */

.card {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 24px;
    padding: 26px;
    box-shadow: 0 8px 30px rgba(16,24,40,.035);
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


/* MODEL CARDS */

.model-card {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 22px;
    padding: 24px;
    min-height: 170px;
}

.model-icon {
    font-size: 30px;
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


/* RESULT */

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


/* MARKETPLACE */

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


/* WARNING */

.warning {
    background: #FFF8E7;
    border: 1px solid #F2CC72;
    color: #694B00;
    border-radius: 15px;
    padding: 15px;
    font-size: 12px;
    line-height: 1.6;
}


/* SECTION */

.section {
    font-size: 25px;
    font-weight: 800;
    color: #101828;
    letter-spacing: -.8px;
    margin-top: 38px;
    margin-bottom: 18px;
}


/* FOOTER */

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
# NAVIGATION
# ================================================================

def navigate(page):

    st.session_state.page = page
    st.rerun()


# ================================================================
# HUGGING FACE MODEL DISCOVERY
# ================================================================

@st.cache_data(ttl=3600)
def find_model_file():

    api = HfApi()

    files = api.list_repo_files(
        repo_id=HF_REPO,
        repo_type="model",
    )

    model_files = [
        f
        for f in files
        if f.lower().endswith(
            (".pt", ".pth", ".bin")
        )
    ]

    if not model_files:

        raise RuntimeError(
            "No PyTorch model file was found "
            "in the Hugging Face repository."
        )

    # Prefer the actual GAIA/MammoSense file

    preferred = [
        f
        for f in model_files
        if (
            "gaia_busi" in f.lower()
            or "mammosense" in f.lower()
        )
    ]

    if preferred:
        return preferred[0]

    return model_files[0]


# ================================================================
# LOAD MAMMOSENSE
# ================================================================

@st.cache_resource(
    show_spinner=False
)
def load_mammosense():

    # ------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # ------------------------------------------------------------
    # FIND MODEL
    # ------------------------------------------------------------

    model_file = find_model_file()


    # ------------------------------------------------------------
    # DOWNLOAD MODEL
    # ------------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=model_file,
        repo_type="model",
    )


    # ------------------------------------------------------------
    # PYTORCH 2.6 COMPATIBILITY
    #
    # weights_only=False is intentional because this is
    # your trusted checkpoint and it contains metadata such
    # as class_names and architecture.
    # ------------------------------------------------------------

    try:

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        # Compatibility with older PyTorch

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

    except Exception as e:

        raise RuntimeError(
            "Unable to load the MammoSense checkpoint.\n\n"
            + str(e)
        )


    # ------------------------------------------------------------
    # CHECKPOINT FORMAT
    # ------------------------------------------------------------

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "The Hugging Face file is not a valid "
            "MammoSense checkpoint."
        )


    # ------------------------------------------------------------
    # READ ARCHITECTURE
    # ------------------------------------------------------------

    architecture = checkpoint.get(
        "architecture",
        "vit_small_patch16_224",
    )


    # ------------------------------------------------------------
    # READ NUMBER OF CLASSES
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # READ CLASS NAMES
    # ------------------------------------------------------------

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

        classes = [
            "Normal",
            "Benign",
            "Malignant",
        ]

        classes = classes[
            :num_classes
        ]


    # ------------------------------------------------------------
    # GET STATE DICTIONARY
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

        # Some checkpoints are already
        # pure state dictionaries.

        state_dict = {
            k: v
            for k, v in checkpoint.items()
            if torch.is_tensor(v)
        }


    if not state_dict:

        raise RuntimeError(
            "No model_state_dict or state_dict "
            "was found in the checkpoint."
        )


    # ------------------------------------------------------------
    # CLEAN STATE DICTIONARY
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # CREATE EXACT ARCHITECTURE
    # ------------------------------------------------------------

    try:

        model = timm.create_model(
            architecture,
            pretrained=False,
            num_classes=num_classes,
            img_size=IMAGE_SIZE,
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not create model architecture:\n\n"
            f"{architecture}\n\n"
            f"{e}"
        )


    # ------------------------------------------------------------
    # LOAD WEIGHTS
    # ------------------------------------------------------------

    try:

        model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

    except RuntimeError as e:

        # Produce useful diagnostic information

        expected_keys = set(
            model.state_dict().keys()
        )

        actual_keys = set(
            cleaned_state_dict.keys()
        )

        missing = list(
            expected_keys - actual_keys
        )[:15]

        unexpected = list(
            actual_keys - expected_keys
        )[:15]

        raise RuntimeError(
            "MammoSense checkpoint does not match "
            "the expected ViT architecture.\n\n"

            f"Architecture: {architecture}\n"
            f"Classes: {classes}\n\n"

            f"Missing keys:\n{missing}\n\n"

            f"Unexpected keys:\n{unexpected}\n\n"

            f"Original error:\n{e}"
        )


    # ------------------------------------------------------------
    # MOVE MODEL TO DEVICE
    # ------------------------------------------------------------

    model = model.to(
        device
    )

    model.eval()


    # ------------------------------------------------------------
    # IMAGE PREPROCESSING
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
        architecture,
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


    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    logits = model(
        tensor
    )


    # ------------------------------------------------------------
    # PROBABILITIES
    # ------------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]


    # ------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------

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


# ================================================================
# HEADER
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


# ================================================================
# NAVIGATION BAR
# ================================================================

n1, n2, n3, n4, n5 = st.columns(
    5
)


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


st.markdown(
    "---"
)


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
                Medusa combines artificial
                intelligence, health insights
                and healthcare services in one
                intelligent platform.
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

        navigate(
            "Detection"
        )


    st.markdown(
        '<div class="section">'
        'AI Models'
        '</div>',
        unsafe_allow_html=True,
    )


    a, b, c = st.columns(
        3
    )


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
                    AI classification.
                </div>

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
                    More medical AI models
                    coming to the platform.
                </div>

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
                Upload a breast ultrasound
                image and let MammoSense
                analyse it.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------

    with st.spinner(
        "Loading MammoSense AI..."
    ):

        try:

            (
                model,
                transform,
                classes,
                device,
                model_file,
                architecture,
            ) = load_mammosense()

            model_ready = True

        except Exception as error:

            model_ready = False

            st.error(
                "MammoSense could not be loaded."
            )

            st.code(
                str(error)
            )


    # ------------------------------------------------------------
    # MODEL INFORMATION
    # ------------------------------------------------------------

    if model_ready:

        st.success(
            "MammoSense is ready."
        )

        with st.expander(
            "Model information"
        ):

            st.write(
                f"**Model:** {model_file}"
            )

            st.write(
                f"**Architecture:** "
                f"{architecture}"
            )

            st.write(
                f"**Classes:** "
                f"{', '.join(classes)}"
            )

            st.write(
                f"**Device:** "
                f"{device}"
            )


        # --------------------------------------------------------
        # IMAGE UPLOAD
        # --------------------------------------------------------

        uploaded = st.file_uploader(
            "Upload breast ultrasound image",
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
                    "The uploaded image could "
                    "not be read."
                )

                st.stop()


            left, right = st.columns(
                [1.2, 1]
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
                            Image ready
                        </div>

                        <br>

                        <div class="card-text">
                            MammoSense will analyse
                            this image using the
                            trained ViT model.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                st.write(
                    f"**File:** {uploaded.name}"
                )

                st.write(
                    f"**Size:** "
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


        st.markdown(
            "---"
        )


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
                    Confidence:
                    {result["confidence"] * 100:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        st.markdown(
            '<div class="section">'
            'Probability Distribution'
            '</div>',
            unsafe_allow_html=True,
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
            'Recommended Next Steps'
            '</div>',
            unsafe_allow_html=True,
        )


        c1, c2, c3 = st.columns(
            3
        )


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
            Medusa provides AI-assisted
            screening support. This result
            is not a diagnosis and should be
            reviewed by a qualified healthcare
            professional.

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
                View your Medusa AI analysis
                history.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    if not st.session_state.history:

        st.info(
            "No AI analysis has been performed yet."
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
                        {item["confidence"] * 100:.2f}%

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
                Discover healthcare professionals,
                diagnostic centres and healthcare
                services through Medusa.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    a, b, c = st.columns(
        3
    )


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
                    Find imaging centres,
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


    st.info(
        "The Medusa marketplace will be "
        "connected to real providers in the "
        "next development phase."
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

                Account management, privacy,
                notifications and health
                preferences will appear here.

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

        MEDUSA AI
        <br>
        Intelligent Health Infrastructure

        <br><br>

        AI-assisted screening only.
        Not a substitute for professional
        medical advice.

    </div>
    """,
    unsafe_allow_html=True,
)
