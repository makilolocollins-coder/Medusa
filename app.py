import streamlit as st

from ui.styles import load_styles
from ui.home import show_home
from ui.health import show_health
from ui.marketplace import show_marketplace
from ui.profile import show_profile
from ui.detection import show_detection


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
# DESIGN
# ============================================================

load_styles()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ============================================================
# HEADER
# ============================================================

top_left, top_right = st.columns([4, 1])

with top_left:

    st.title("🧬 MEDUSA AI")
    st.caption("Intelligent Health Infrastructure")

with top_right:

    st.success("● AI ONLINE")


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

selected_page = st.radio(
    "Navigation",
    pages,
    index=pages.index(
        st.session_state.page
    ),
    horizontal=True,
    label_visibility="collapsed",
)

st.session_state.page = selected_page


st.divider()


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
