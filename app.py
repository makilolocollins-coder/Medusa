# ================================================================
# MEDUSA AI
# PROFESSIONAL MEDICAL PLATFORM
#
# ROLE-BASED NAVIGATION
# PATIENT PORTAL + RADIOLOGIST PORTAL
# ================================================================

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

from utils.supabase_client import get_supabase


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Medusa AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# GLOBAL DESIGN
# ================================================================

load_styles()


# ================================================================
# SESSION STATE
# ================================================================

DEFAULT_STATE = {

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

    "patient_id": None,

    "patient_name": "",

    "patient_state": "",

}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ================================================================
# SUPABASE
# ================================================================

supabase = get_supabase()


# ================================================================
# AUTHENTICATION
# ================================================================

if not st.session_state.authenticated:

    # ------------------------------------------------------------
    # REGISTER
    # ------------------------------------------------------------

    if st.session_state.auth_step == "register":

        show_auth()

        st.divider()

        if st.button(
            "Already have an account? Sign in",
            use_container_width=True,
        ):

            st.session_state.auth_step = "login"

            st.rerun()


    # ------------------------------------------------------------
    # VERIFY
    # ------------------------------------------------------------

    elif st.session_state.auth_step == "verify":

        show_verification()

        st.divider()

        if st.button(
            "Back to sign in",
            use_container_width=True,
        ):

            st.session_state.auth_step = "login"

            st.rerun()


    # ------------------------------------------------------------
    # LOGIN
    # ------------------------------------------------------------

    else:

        show_login()

        st.divider()

        if st.button(
            "Create an account",
            use_container_width=True,
        ):

            st.session_state.auth_step = "register"

            st.rerun()


    st.stop()


# ================================================================
# CURRENT USER
# ================================================================

current_user = None

is_radiologist = False

radiologist_profile = None


try:

    response = supabase.auth.get_user()

    if response.user:

        current_user = response.user

except Exception:

    current_user = None


# ================================================================
# VERIFY SESSION
# ================================================================

if current_user is None:

    st.session_state.authenticated = False

    st.session_state.auth_step = "login"

    st.rerun()


# ================================================================
# CHECK RADIOLOGIST ROLE
# ================================================================

try:

    doctor_response = (
        supabase
        .table("radiologists")
        .select(
            "user_id, full_name, active"
        )
        .eq(
            "user_id",
            current_user.id,
        )
        .eq(
            "active",
            True,
        )
        .limit(1)
        .execute()
    )

    doctor_data = (
        doctor_response.data
        or []
    )

    if doctor_data:

        is_radiologist = True

        radiologist_profile = (
            doctor_data[0]
        )

except Exception:

    is_radiologist = False

    radiologist_profile = None


# ================================================================
# TOP BRAND HEADER
# ================================================================

st.markdown(
    """
    <div class="medusa-header">

        <div class="medusa-brand">
            <div class="medusa-logo">
                M
            </div>

            <div>
                <div class="medusa-name">
                    MEDUSA
                </div>

                <div class="medusa-subtitle">
                    Intelligent Health Infrastructure
                </div>
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    # ------------------------------------------------------------
    # BRAND
    # ------------------------------------------------------------

    st.markdown(
        """
        <div style="
            padding: 10px 0 20px 0;
        ">

            <div style="
                font-size: 24px;
                font-weight: 800;
                letter-spacing: 1px;
            ">
                MEDUSA
            </div>

            <div style="
                font-size: 12px;
                opacity: 0.65;
                margin-top: 3px;
            ">
                Medical Intelligence Platform
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ------------------------------------------------------------
    # USER CARD
    # ------------------------------------------------------------

    email = (
        current_user.email
        if current_user
        else ""
    )

    if is_radiologist:

        role_label = "RADIOLOGIST"

        role_icon = "👨‍⚕️"

        display_name = (
            radiologist_profile.get(
                "full_name"
            )
            if radiologist_profile
            else "Radiologist"
        )

    else:

        role_label = "PATIENT"

        role_icon = "👤"

        display_name = (
            email.split("@")[0]
            if email
            else "Patient"
        )


    st.markdown(
        f"""
        <div style="
            padding: 14px;
            border-radius: 14px;
            background: rgba(255,255,255,0.06);
            margin-bottom: 20px;
        ">

            <div style="
                font-size: 20px;
            ">
                {role_icon}
            </div>

            <div style="
                font-weight: 700;
                margin-top: 5px;
            ">
                {display_name}
            </div>

            <div style="
                font-size: 11px;
                opacity: 0.6;
                margin-top: 3px;
                letter-spacing: 1px;
            ">
                {role_label}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ------------------------------------------------------------
    # NAVIGATION TITLE
    # ------------------------------------------------------------

    st.caption(
        "MAIN MENU"
    )


    # ============================================================
    # PATIENT NAVIGATION
    # ============================================================

    if not is_radiologist:

        patient_pages = {

            "Dashboard": "🏠",

            "AI Detection": "🔬",

            "My Examinations": "📋",

            "Radiologist Reviews": "👨‍⚕️",

            "Consultations": "📅",

            "Health": "❤️",

            "Marketplace": "🛒",

            "Profile": "👤",

        }


        for page_name, icon in patient_pages.items():

            active = (
                st.session_state.page
                == page_name
            )

            label = (
                f"{icon}  {page_name}"
            )

            if st.button(
                label,
                key=f"nav_{page_name}",
                use_container_width=True,
                type=(
                    "primary"
                    if active
                    else "secondary"
                ),
            ):

                st.session_state.page = (
                    page_name
                )

                st.rerun()


    # ============================================================
    # RADIOLOGIST NAVIGATION
    # ============================================================

    else:

        radiologist_pages = {

            "Dashboard": "🏠",

            "Review Queue": "📥",

            "Patient Examinations": "👥",

            "Approved Reports": "📄",

            "Profile": "👤",

        }


        for page_name, icon in radiologist_pages.items():

            active = (
                st.session_state.page
                == page_name
            )

            label = (
                f"{icon}  {page_name}"
            )

            if st.button(
                label,
                key=f"nav_doctor_{page_name}",
                use_container_width=True,
                type=(
                    "primary"
                    if active
                    else "secondary"
                ),
            ):

                st.session_state.page = (
                    page_name
                )

                st.rerun()


    # ------------------------------------------------------------
    # SIDEBAR FOOTER
    # ------------------------------------------------------------

    st.divider()

    st.caption(
        "MEDUSA AI"
    )

    st.caption(
        "AI-assisted clinical screening"
    )


    # ------------------------------------------------------------
    # SIGN OUT
    # ------------------------------------------------------------

    if st.button(
        "↪ Sign out",
        use_container_width=True,
    ):

        try:

            supabase.auth.sign_out()

        except Exception:

            pass

        # Clear authentication-related state

        st.session_state.authenticated = False

        st.session_state.auth_step = "login"

        st.session_state.page = "Dashboard"

        st.session_state.scan_result = None

        st.session_state.scan_id = None

        st.session_state.patient_id = None

        st.rerun()


# ================================================================
# MAIN CONTENT HEADER
# ================================================================

page = st.session_state.page


# ================================================================
# PATIENT ROUTING
# ================================================================

if not is_radiologist:

    if page == "Dashboard":

        show_home()


    elif page == "AI Detection":

        show_detection()


    elif page == "My Examinations":

        st.title(
            "📋 My Examinations"
        )

        st.info(
            "Your examination history will appear here."
        )


    elif page == "Radiologist Reviews":

        st.title(
            "👨‍⚕️ Radiologist Reviews"
        )

        st.info(
            "Your submitted examinations and "
            "radiologist review status will appear here."
        )


    elif page == "Consultations":

        st.title(
            "📅 Consultations"
        )

        st.info(
            "Your radiologist consultations "
            "will appear here."
        )


    elif page == "Health":

        show_health()


    elif page == "Marketplace":

        show_marketplace()


    elif page == "Profile":

        show_profile()


# ================================================================
# RADIOLOGIST ROUTING
# ================================================================

else:

    if page == "Dashboard":

        st.title(
            "👨‍⚕️ Radiologist Dashboard"
        )

        st.caption(
            "Clinical review and reporting workspace"
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Pending Reviews",
                "—",
            )

        with col2:

            st.metric(
                "Completed Reviews",
                "—",
            )

        with col3:

            st.metric(
                "Approved Reports",
                "—",
            )

        st.divider()

        st.info(
            "Select Review Queue to review "
            "patient examinations."
        )


    elif page == "Review Queue":

        show_radiologist()


    elif page == "Patient Examinations":

        st.title(
            "👥 Patient Examinations"
        )

        st.info(
            "Patient examinations will appear here."
        )


    elif page == "Approved Reports":

        st.title(
            "📄 Approved Reports"
        )

        st.info(
            "Approved medical reports will appear here."
        )


    elif page == "Profile":

        show_profile()


# ================================================================
# SECURITY NOTICE
# ================================================================

st.divider()

if is_radiologist:

    st.caption(
        "🔒 Radiologist workspace • "
        "Clinical review and reporting"
    )

else:

    st.caption(
        "🔒 Patient portal • "
        "AI-assisted screening"
    )


st.caption(
    "MEDUSA AI • AI-assisted screening only. "
    "Not a substitute for professional medical diagnosis."
)
