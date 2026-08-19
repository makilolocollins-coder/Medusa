import streamlit as st

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
# LOAD DESIGN
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

left, right = st.columns([4, 1])

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


columns = st.columns(len(pages))


for column, page in zip(columns, pages):

    with column:

        if st.button(
            page,
            use_container_width=True,
        ):

            st.session_state.page = page

            st.rerun()


st.markdown("---")


# ============================================================
# ROUTING
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
