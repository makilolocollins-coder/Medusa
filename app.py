import streamlit as st

from ui.styles import load_styles
from ui.home import show_home
from ui.detection import show_detection
from ui.health import show_health
from ui.marketplace import show_marketplace
from ui.profile import show_profile
from ui.radiologist import show_radiologist

from ui.auth import (
    show_auth,
    show_verification,
    show_login,
)


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

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "auth_step" not in st.session_state:
    st.session_state.auth_step = "login"

if "auth_email" not in st.session_state:
    st.session_state.auth_email = ""

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

if "scan_id" not in st.session_state:
    st.session_state.scan_id = None

if "consultation_booked" not in st.session_state:
    st.session_state.consultation_booked = False

if "review_requested" not in st.session_state:
    st.session_state.review_requested = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ============================================================
# AUTHENTICATION
# ============================================================

if not st.session_state.authenticated:

    # --------------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------------

    if st.session_state.auth_step == "register":

        show_auth()

        st.divider()

        if st.button(
            "Already have an account? Login",
            use_container_width=True,
        ):

            st.session_state.auth_step = "login"
            st.rerun()


    # --------------------------------------------------------
    # VERIFICATION
    # --------------------------------------------------------

    elif st.session_state.auth_step == "verify":

        show_verification()

        st.divider()

        if st.button(
            "Back to login",
            use_container_width=True,
        ):

            st.session_state.auth_step = "login"
            st.rerun()


    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    else:

        show_login()

        st.divider()

        if st.button(
            "Create a new account",
            use_container_width=True,
        ):

            st.session_state.auth_step = "register"
            st.rerun()


    st.stop()


# ============================================================
# MAIN APPLICATION
# ============================================================

st.markdown(
    """
    <div class="brand">
        MEDUSA<span>◉</span>
    </div>
    """,
    unsafe_allow_html=True,
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
