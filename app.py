# ============================================================
# MEDUSA AI
# MAIN APPLICATION
# ============================================================

import streamlit as st

from ui.styles import load_styles

from ui.home import show_home
from ui.dashboard import show_dashboard
from ui.detection import show_detection
from ui.health import show_health
from ui.marketplace import show_marketplace
from ui.profile import show_profile
from ui.radiologist import show_radiologist

from reports.pdf_reports_page import show_pdf_reports

from ui.auth import (
    show_auth,
    show_verification,
    show_login,
)

from utils.supabase_client import get_supabase


# ============================================================
# PAGE CONFIGURATION
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

defaults = {
    "authenticated": False,
    "auth_step": "login",
    "auth_email": "",
    "page": "Dashboard",
    "history": [],
    "scan_result": None,
    "scan_id": None,
    "consultation_booked": False,
    "review_requested": False,
    "prediction": None,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SUPABASE
# ============================================================

supabase = get_supabase()

current_user = None
is_radiologist = False


# ============================================================
# CHECK AUTHENTICATION
# ============================================================

if st.session_state.authenticated:

    try:

        response = supabase.auth.get_user()

        if response.user:

            current_user = response.user

            doctor = (
                supabase
                .table("radiologists")
                .select("user_id, full_name")
                .eq(
                    "user_id",
                    current_user.id,
                )
                .eq(
                    "active",
                    True,
                )
                .execute()
                .data
                or []
            )

            is_radiologist = bool(doctor)

        else:

            st.session_state.authenticated = False

    except Exception:

        current_user = None
        is_radiologist = False


# ============================================================
# AUTHENTICATION SCREEN
# ============================================================

if not st.session_state.authenticated:

    if st.session_state.auth_step == "register":

        show_auth()

        st.divider()

        if st.button(
            "Already have an account? Login",
            use_container_width=True,
        ):

            st.session_state.auth_step = "login"
            st.rerun()

    elif st.session_state.auth_step == "verify":

        show_verification()

        st.divider()

        if st.button(
            "Back to login",
            use_container_width=True,
        ):

            st.session_state.auth_step = "login"
            st.rerun()

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
# BRAND
# ============================================================

st.title("MEDUSA")

st.caption(
    "Intelligent Health Infrastructure"
)


# ============================================================
# PATIENT NAVIGATION
# ============================================================

patient_pages = [
    "Dashboard",
    "AI Detection",
    "Examinations",
    "Reports",
    "Health",
    "Marketplace",
    "Profile",
]


# ============================================================
# RADIOLOGIST NAVIGATION
# ============================================================

if is_radiologist:

    pages = [
        "Dashboard",
        "AI Detection",
        "Examinations",
        "Reports",
        "Health",
        "Marketplace",
        "Profile",
        "Radiologist",
    ]

else:

    pages = patient_pages


# ============================================================
# NAVIGATION
# ============================================================

selected_page = st.radio(
    "Main navigation",
    pages,
    index=(
        pages.index(st.session_state.page)
        if st.session_state.page in pages
        else 0
    ),
    horizontal=True,
    label_visibility="collapsed",
)

st.session_state.page = selected_page


# ============================================================
# PAGE ROUTING
# ============================================================

if selected_page == "Dashboard":

    show_dashboard()


elif selected_page == "AI Detection":

    show_detection()


elif selected_page == "Examinations":

    st.title("Examinations")

    st.info(
        "Your examination history is available "
        "on the Dashboard."
    )


elif selected_page == "Reports":

    show_pdf_reports()


elif selected_page == "Health":

    show_health()


elif selected_page == "Marketplace":

    show_marketplace()


elif selected_page == "Profile":

    show_profile()


elif selected_page == "Radiologist":

    if is_radiologist:

        show_radiologist()

    else:

        st.error(
            "Unauthorized access."
        )


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
