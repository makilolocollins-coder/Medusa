# ============================================================
# MEDUSA AI
# MAIN APPLICATION
# STANDARD MEDICAL NAVIGATION
# ============================================================

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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medusa AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL DESIGN
# ============================================================

load_styles()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "authenticated": False,
    "auth_step": "login",
    "auth_email": "",
    "page": "Dashboard",

    "history": [],

    "scan_result": None,
    "scan_id": None,
    "scan_model": None,
    "scan_type": None,

    "consultation_booked": False,
    "review_requested": False,

    "prediction": None,

    "patient_id": None,
    "patient_name": "",
    "patient_state": "",

    "scan_image_bytes": None,
    "scan_filename": None,

    "review_status": "Not requested",
    "review_id": None,

    "report_pdf": None,
    "report_id": None,
    "report_downloadable": False,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# SUPABASE
# ============================================================

supabase = get_supabase()


# ============================================================
# AUTHENTICATION STATE
# ============================================================

current_user = None
is_radiologist = False
radiologist_name = None


if st.session_state.authenticated:

    try:

        response = supabase.auth.get_user()

        if response.user:

            current_user = response.user

            # ------------------------------------------------
            # CHECK WHETHER USER IS A RADIOLOGIST
            # ------------------------------------------------

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

            doctors = (
                doctor_response.data
                or []
            )

            if doctors:

                is_radiologist = True

                radiologist_name = (
                    doctors[0].get(
                        "full_name"
                    )
                )

    except Exception:

        current_user = None
        is_radiologist = False


# ============================================================
# AUTHENTICATION UI
# ============================================================

if not st.session_state.authenticated:

    # --------------------------------------------------------
    # REGISTER
    # --------------------------------------------------------

    if (
        st.session_state.auth_step
        == "register"
    ):

        show_auth()

        st.divider()

        if st.button(
            "Already have an account? Login",
            use_container_width=True,
        ):

            st.session_state.auth_step = "login"

            st.rerun()

    # --------------------------------------------------------
    # EMAIL VERIFICATION
    # --------------------------------------------------------

    elif (
        st.session_state.auth_step
        == "verify"
    ):

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
# MEDUSA BRAND HEADER
# ============================================================

st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding:10px 0 20px 0;
    ">

        <div style="
            font-size:28px;
            font-weight:800;
            letter-spacing:-1px;
        ">
            MEDUSA<span style="
                font-size:24px;
                margin-left:4px;
            ">◉</span>
        </div>

        <div style="
            font-size:13px;
            opacity:0.65;
        ">
            Intelligent Health Infrastructure
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:800;
            margin-bottom:5px;
        ">
            MEDUSA<span>◉</span>
        </div>

        <div style="
            font-size:12px;
            opacity:0.65;
            margin-bottom:25px;
        ">
            Medical Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    if current_user:

        display_email = (
            current_user.email
            or "Authenticated user"
        )

        st.caption(
            f"Signed in as\n{display_email}"
        )

    st.divider()

    # ========================================================
    # PATIENT NAVIGATION
    # ========================================================

    st.markdown(
        "**MAIN**"
    )

    patient_pages = {
        "🏠 Dashboard": "Dashboard",
        "🔬 AI Detection": "AI Detection",
        "📄 My Reports": "My Reports",
        "❤️ Health": "Health",
        "🛒 Marketplace": "Marketplace",
        "👤 Profile": "Profile",
    }

    for label, page_name in patient_pages.items():

        if st.button(
            label,
            key=f"nav_{page_name}",
            use_container_width=True,
        ):

            st.session_state.page = page_name

            st.rerun()

    # ========================================================
    # RADIOLOGIST NAVIGATION
    # ========================================================

    if is_radiologist:

        st.divider()

        st.markdown(
            "**CLINICAL WORKSPACE**"
        )

        if st.button(
            "🩺 Radiologist Workspace",
            key="nav_radiologist",
            use_container_width=True,
        ):

            st.session_state.page = (
                "Radiologist"
            )

            st.rerun()

    # ========================================================
    # ACCOUNT
    # ========================================================

    st.divider()

    st.markdown(
        "**ACCOUNT**"
    )

    if st.button(
        "🚪 Sign out",
        key="sign_out",
        use_container_width=True,
    ):

        try:

            supabase.auth.sign_out()

        except Exception:

            pass

        st.session_state.authenticated = False

        st.session_state.auth_step = "login"

        st.session_state.page = "Dashboard"

        st.rerun()


# ============================================================
# CURRENT PAGE
# ============================================================

current_page = (
    st.session_state.page
)


# ============================================================
# DASHBOARD
# ============================================================

if current_page == "Dashboard":

    show_home()


# ============================================================
# AI DETECTION
# ============================================================

elif current_page == "AI Detection":

    show_detection()


# ============================================================
# HEALTH
# ============================================================

elif current_page == "Health":

    show_health()


# ============================================================
# MARKETPLACE
# ============================================================

elif current_page == "Marketplace":

    show_marketplace()


# ============================================================
# PROFILE
# ============================================================

elif current_page == "Profile":

    show_profile()


# ============================================================
# RADIOLOGIST
# ============================================================

elif current_page == "Radiologist":

    if is_radiologist:

        show_radiologist()

    else:

        st.error(
            "Access denied."
        )

        st.info(
            "Radiologist access is restricted "
            "to verified clinical accounts."
        )


# ============================================================
# MY REPORTS
# ============================================================

elif current_page == "My Reports":

    st.title(
        "📄 My Medical Reports"
    )

    st.caption(
        "View reports that have been reviewed "
        "and approved by a radiologist."
    )

    if current_user is None:

        st.error(
            "Please log in again."
        )

        st.stop()

    try:

        reports_response = (
            supabase
            .table("medical_reports")
            .select(
                """
                report_id,
                patient_id,
                patient_name,
                patient_state,
                status,
                approved_at,
                pdf_path
                """
            )
            .eq(
                "user_id",
                current_user.id,
            )
            .eq(
                "status",
                "APPROVED",
            )
            .order(
                "approved_at",
                desc=True,
            )
            .execute()
        )

        reports = (
            reports_response.data
            or []
        )

    except Exception as error:

        st.error(
            "Unable to load your reports."
        )

        st.exception(error)

        reports = []

    if not reports:

        st.info(
            "No approved medical reports yet."
        )

    else:

        for report in reports:

            with st.container(
                border=True
            ):

                col1, col2 = st.columns(
                    [3, 1]
                )

                with col1:

                    st.subheader(
                        report.get(
                            "report_id",
                            "Medical Report",
                        )
                    )

                    st.write(
                        f"**Patient:** "
                        f"{report.get('patient_name', 'N/A')}"
                    )

                    st.write(
                        f"**Patient ID:** "
                        f"{report.get('patient_id', 'N/A')}"
                    )

                    st.write(
                        f"**State:** "
                        f"{report.get('patient_state', 'N/A')}"
                    )

                    st.caption(
                        f"Approved: "
                        f"{report.get('approved_at', 'N/A')}"
                    )

                with col2:

                    # ------------------------------------------------
                    # SECURE REPORT DOWNLOAD
                    # ------------------------------------------------

                    if report.get("pdf_path"):

                        try:

                            file_response = (
                                supabase
                                .storage
                                .from_(
                                    "medical-reports"
                                )
                                .download(
                                    report[
                                        "pdf_path"
                                    ]
                                )
                            )

                            st.download_button(
                                "⬇️ Download",
                                data=file_response,
                                file_name=(
                                    f"{report['report_id']}.pdf"
                                ),
                                mime="application/pdf",
                                use_container_width=True,
                                key=(
                                    f"download_"
                                    f"{report['report_id']}"
                                ),
                            )

                        except Exception:

                            st.warning(
                                "Report file unavailable."
                            )


# ============================================================
# UNKNOWN PAGE SAFETY
# ============================================================

else:

    st.session_state.page = "Dashboard"

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MEDUSA AI • Intelligent Health Infrastructure"
)

st.caption(
    "AI-assisted screening only. "
    "Final clinical interpretation requires "
    "qualified medical review."
)
