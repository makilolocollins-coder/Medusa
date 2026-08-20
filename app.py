import streamlit as st
from pathlib import Path
import base64

from ui.styles import load_styles
from ui.home import show_home
from ui.detection import show_detection
from ui.health import show_health
from ui.marketplace import show_marketplace
from ui.profile import show_profile


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Medusa AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# BACKGROUND
# ============================================================

def set_background():

    image_path = Path(__file__).parent / "background.jpg"

    if not image_path.exists():
        return

    with open(image_path, "rb") as image:

        encoded = base64.b64encode(
            image.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(247,248,250,0.88),
                    rgba(247,248,250,0.94)
                ),
                url("data:image/jpeg;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


set_background()


# ============================================================
# DESIGN
# ============================================================

load_styles()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"


# ============================================================
# HEADER
# ============================================================

st.title("🧬 MEDUSA AI")

st.caption(
    "Intelligent Health Infrastructure"
)


# ============================================================
# NAVIGATION
# ============================================================

pages = [
    "Home",
    "AI Detection",
    "Health",
    "Marketplace",
    "Profile",
]

selected = st.radio(
    "Navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed",
)

st.session_state.page = selected


# ============================================================
# PAGE ROUTING
# ============================================================

if st.session_state.page == "Home":

    show_home()


elif st.session_state.page == "AI Detection":

    show_detection()


elif st.session_state.page == "Health":

    show_health()


elif st.session_state.page == "Marketplace":

    show_marketplace()


elif st.session_state.page == "Profile":

    show_profile()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MEDUSA AI • Intelligent Health Infrastructure"
)

st.caption(
    "AI-assisted screening only. "
    "Not a substitute for professional medical advice."
)
