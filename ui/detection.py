# ================================================================
# MEDUSA AI
# PATIENT DETECTION WORKFLOW
#
# PATIENT:
#   Name -> State -> Patient ID -> Scan -> AI -> Review Request
#
# IMPORTANT:
#   Patient NEVER enters radiologist findings/impression/etc.
#   PDF remains LOCKED until radiologist approval.
# ================================================================
# ================================================================
# MEDUSA WORKFLOW TEST FLAG
# ================================================================

MEDUSA_WORKFLOW_TEST = True

# True:
#   Shows diagnostic information so we can confirm
#   which workflow/version is actually running.
#
# False:
#   Normal production mode.
# ================================================================

import io
import uuid
from datetime import datetime

import streamlit as st
from PIL import Image

from ui.background import set_background
from utils.supabase_client import get_supabase

from ai.mammosense import (
    load_model as load_mammo_model,
    predict as predict_mammo,
)

from ai.pneumonia import (
    load_model as load_pneumonia_model,
    predict as predict_pneumonia,
)


# ================================================================
# NIGERIAN STATES
# ================================================================

NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra",
    "Bauchi", "Bayelsa", "Benue", "Borno",
    "Cross River", "Delta", "Ebonyi", "Edo",
    "Ekiti", "Enugu", "Gombe", "Imo",
    "Jigawa", "Kaduna", "Kano", "Katsina",
    "Kebbi", "Kogi", "Kwara", "Lagos",
    "Nasarawa", "Niger", "Ogun", "Ondo",
    "Osun", "Oyo", "Plateau", "Rivers",
    "Sokoto", "Taraba", "Yobe", "Zamfara",
    "FCT",
]


# ================================================================
# SESSION STATE
# ================================================================

def initialize_state():

    defaults = {
        "patient_id": None,
        "patient_name": "",
        "patient_state": "Delta",

        "scan_id": None,
        "scan_model": None,
        "scan_type": None,
        "scan_result": None,
        "scan_image_bytes": None,
        "scan_filename": None,

        "review_status": None,
        "review_id": None,

        "report_id": None,
        "report_pdf": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ================================================================
# PATIENT ID
# ================================================================

def generate_patient_id():

    date_part = datetime.now().strftime("%Y%m%d")

    random_part = uuid.uuid4().hex[:8].upper()

    return f"MED-P-{date_part}-{random_part}"


# ================================================================
# USER
# ================================================================

def get_current_user():

    supabase = get_supabase()

    response = supabase.auth.get_user()

    if not response.user:
        return None

    return response.user


# ================================================================
# RESET
# ================================================================

def reset_scan():

    st.session_state.scan_id = None
    st.session_state.scan_model = None
    st.session_state.scan_type = None
    st.session_state.scan_result = None
    st.session_state.scan_image_bytes = None
    st.session_state.scan_filename = None

    st.session_state.review_status = None
    st.session_state.review_id = None

    st.session_state.report_id = None
    st.session_state.report_pdf = None


# ================================================================
# SHOW DETECTION
# ================================================================

def show_detection():

    initialize_state()

    set_background("detection.jpg")

    st.title("🧬 Medusa AI")

    st.caption(
        "AI-assisted medical image screening"
    )

    # ============================================================
    # PATIENT REGISTRATION
    # ============================================================

    st.subheader("Patient Registration")

    col1, col2 = st.columns(2)

    with col1:

        patient_name = st.text_input(
            "Patient full name",
            value=st.session_state.patient_name,
            placeholder="Enter full name",
            key="patient_name_field",
        ).strip()

    with col2:

        current_state = st.session_state.patient_state

        state_index = (
            NIGERIAN_STATES.index(current_state)
            if current_state in NIGERIAN_STATES
            else 0
        )

        patient_state = st.selectbox(
            "State",
            NIGERIAN_STATES,
            index=state_index,
            key="patient_state_field",
        )

    st.session_state.patient_name = patient_name
    st.session_state.patient_state = patient_state

    # ============================================================
    # PATIENT ID
    # ============================================================

    if patient_name and not st.session_state.patient_id:

        st.session_state.patient_id = (
            generate_patient_id()
        )

    if st.session_state.patient_id:

        st.success(
            f"Patient ID: "
            f"**{st.session_state.patient_id}**"
        )

    # ============================================================
    # MODEL
    # ============================================================

    st.subheader("Examination")

    model_choice = st.selectbox(
        "Select examination",
        [
            "MammoSense — Breast Ultrasound",
            "MammoSense Pneumonia — Chest X-ray",
        ],
        key="patient_model_choice",
    )

    is_pneumonia = (
        model_choice
        == "MammoSense Pneumonia — Chest X-ray"
    )

    if is_pneumonia:

        examination = "Chest X-ray"
        model_name = "MammoSense Pneumonia V2"

        st.info(
            "🫁 Upload a chest X-ray for "
            "AI-assisted pneumonia screening."
        )

    else:

        examination = "Breast Ultrasound"
        model_name = "MammoSense V2"

        st.info(
            "🩻 Upload a breast ultrasound "
            "for AI-assisted screening."
        )

    # ============================================================
    # NEW SCAN
    # ============================================================

    if st.session_state.scan_result is not None:

        if st.button(
            "＋ Start New Examination",
            use_container_width=True,
        ):

            reset_scan()

            st.rerun()

    # ============================================================
    # UPLOAD
    # ============================================================

    uploaded = st.file_uploader(
        f"Upload {examination}",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="patient_scan_upload",
    )

    if uploaded is not None:

        image_bytes = uploaded.getvalue()

        st.session_state.scan_image_bytes = (
            image_bytes
        )

        st.session_state.scan_filename = (
            uploaded.name
        )

    elif st.session_state.scan_image_bytes:

        image_bytes = (
            st.session_state.scan_image_bytes
        )

    else:

        st.info(
            f"Please upload a {examination.lower()}."
        )

        return

    # ============================================================
    # DISPLAY IMAGE
    # ============================================================

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        st.image(
            image,
            caption=examination,
            use_container_width=True,
        )

    except Exception:

        st.error(
            "The uploaded image could not be opened."
        )

        return

    # ============================================================
    # ANALYZE
    # ============================================================

    if st.button(
        "🔬 Analyze Scan",
        type="primary",
        use_container_width=True,
        key="analyze_scan_button",
    ):

        if not patient_name:

            st.error(
                "Please enter the patient's full name."
            )

            return

        if not st.session_state.patient_id:

            st.session_state.patient_id = (
                generate_patient_id()
            )

        try:

            with st.spinner(
                "Medusa AI is analyzing the scan..."
            ):

                if is_pneumonia:

                    load_pneumonia_model()

                    result = predict_pneumonia(
                        image
                    )

                else:

                    load_mammo_model()

                    result = predict_mammo(
                        image
                    )

            user = get_current_user()

            if user is None:

                st.error(
                    "Your login session has expired."
                )

                return

            supabase = get_supabase()

            # ----------------------------------------------------
            # STORAGE
            # ----------------------------------------------------

            extension = (
                uploaded.name.split(".")[-1].lower()
                if uploaded is not None
                else "png"
            )

            image_path = (
                f"{user.id}/"
                f"{st.session_state.patient_id}/"
                f"{uuid.uuid4().hex}.{extension}"
            )

            supabase.storage.from_(
                "mammosense-scans"
            ).upload(
                image_path,
                image_bytes,
                {
                    "content-type": (
                        uploaded.type
                        if uploaded is not None
                        else "image/png"
                    ),
                    "upsert": "false",
                },
            )

            # ----------------------------------------------------
            # DATABASE
            # ----------------------------------------------------

            response = (
                supabase
                .table("ai_scans")
                .insert({
                    "user_id": user.id,

                    "patient_id":
                        st.session_state.patient_id,

                    "patient_name":
                        patient_name,

                    "patient_state":
                        patient_state,

                    "examination":
                        examination,

                    "model":
                        model_name,

                    "prediction":
                        result["prediction"],

                    "confidence":
                        result["confidence"],

                    "probabilities":
                        result.get(
                            "probabilities",
                            {},
                        ),

                    "image_path":
                        image_path,

                    "status":
                        "AI_COMPLETED",
                })
                .execute()
            )

            if not response.data:

                st.error(
                    "The scan could not be saved."
                )

                return

            scan_id = response.data[0]["id"]

            # ----------------------------------------------------
            # SESSION
            # ----------------------------------------------------

            st.session_state.scan_id = scan_id

            st.session_state.scan_model = model_name

            st.session_state.scan_type = (
                "pneumonia"
                if is_pneumonia
                else "mammosense"
            )

            st.session_state.scan_result = result

            st.session_state.review_status = (
                "NOT_REQUESTED"
            )

            st.session_state.review_id = None

            st.session_state.report_id = None
            st.session_state.report_pdf = None

            st.success(
                "✅ AI analysis completed."
            )

        except Exception as error:

            st.error(
                "The scan could not be analyzed."
            )

            st.exception(error)

    # ============================================================
    # RESULT
    # ============================================================

    result = st.session_state.scan_result

    if result is None:
        return

    st.divider()

    st.subheader(
        "🤖 AI Screening Result"
    )

    prediction = result.get(
        "prediction",
        "Unknown",
    )

    confidence = float(
        result.get(
            "confidence",
            0,
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        if prediction.upper() in (
            "PNEUMONIA",
            "MALIGNANT",
        ):

            st.error(
                f"⚠️ {prediction}"
            )

        else:

            st.success(
                f"✓ {prediction}"
            )

    with col2:

        st.metric(
            "AI Confidence",
            f"{confidence:.1%}",
        )

    # ============================================================
    # PROBABILITIES
    # ============================================================

    probabilities = result.get(
        "probabilities",
        {},
    )

    if probabilities:

        st.subheader(
            "Probability Breakdown"
        )

        for name, value in probabilities.items():

            value = float(value)

            st.write(
                f"**{name}: {value:.2%}**"
            )

            st.progress(
                min(
                    max(value, 0.0),
                    1.0,
                )
            )

    # ============================================================
    # REVIEW STATUS
    # ============================================================

    st.divider()

    st.subheader(
        "👨‍⚕️ Radiologist Review"
    )

    st.info(
        "Every examination must be reviewed and "
        "approved by a qualified radiologist before "
        "a final medical report can be downloaded."
    )

    status = st.session_state.review_status

    # ============================================================
    # REQUEST REVIEW
    # ============================================================

    if status == "NOT_REQUESTED":

        if st.button(
            "📋 Submit for Radiologist Review",
            type="primary",
            use_container_width=True,
            key="submit_review",
        ):

            try:

                user = get_current_user()

                if user is None:

                    st.error(
                        "Please log in again."
                    )

                    return

                supabase = get_supabase()

                review_response = (
                    supabase
                    .table(
                        "radiologist_reviews"
                    )
                    .insert({
                        "scan_id":
                            st.session_state.scan_id,

                        "user_id":
                            user.id,

                        "status":
                            "PENDING",

                        "approved":
                            False,
                    })
                    .execute()
                )

                if not review_response.data:

                    st.error(
                        "The review request "
                        "could not be submitted."
                    )

                    return

                review = review_response.data[0]

                st.session_state.review_id = (
                    review["id"]
                )

                st.session_state.review_status = (
                    "PENDING"
                )

                (
                    supabase
                    .table("ai_scans")
                    .update({
                        "status":
                            "PENDING_REVIEW",
                    })
                    .eq(
                        "id",
                        st.session_state.scan_id,
                    )
                    .execute()
                )

                st.success(
                    "✅ Examination submitted "
                    "for radiologist review."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "Could not submit the examination "
                    "for review."
                )

                st.exception(error)

    # ============================================================
    # PENDING
    # ============================================================

    elif status == "PENDING":

        st.warning(
            "⏳ Awaiting Radiologist Review"
        )

        st.caption(
            "The final medical report remains locked "
            "until the radiologist completes and "
            "approves the review."
        )

    # ============================================================
    # APPROVED
    # ============================================================

    elif status == "APPROVED":

        st.success(
            "✅ Radiologist review completed."
        )

        # --------------------------------------------------------
        # FETCH REPORT
        # --------------------------------------------------------

        try:

            supabase = get_supabase()

            report = (
                supabase
                .table("medical_reports")
                .select("*")
                .eq(
                    "scan_id",
                    st.session_state.scan_id,
                )
                .eq(
                    "status",
                    "APPROVED",
                )
                .limit(1)
                .execute()
                .data
            )

            if report:

                report = report[0]

                st.session_state.report_id = (
                    report.get("report_id")
                )

                pdf_path = report.get(
                    "pdf_path"
                )

                # ------------------------------------------------
                # DOWNLOAD PDF FROM STORAGE
                # ------------------------------------------------

                if pdf_path:

                    pdf_bytes = (
                        supabase.storage
                        .from_("medical-reports")
                        .download(pdf_path)
                    )

                    st.session_state.report_pdf = (
                        pdf_bytes
                    )

        except Exception as error:

            st.warning(
                "The approved report exists, "
                "but could not be loaded."
            )

        # --------------------------------------------------------
        # DOWNLOAD
        # --------------------------------------------------------

        if (
            st.session_state.report_pdf
            and st.session_state.report_id
        ):

            st.success(
                f"Final Report: "
                f"**{st.session_state.report_id}**"
            )

            st.download_button(
                "⬇️ Download Final Medical Report",
                data=st.session_state.report_pdf,
                file_name=(
                    f"{st.session_state.report_id}.pdf"
                ),
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

        else:

            st.info(
                "Your report has been approved and "
                "is being prepared."
            )

    # ============================================================
    # HARD SAFETY MESSAGE
    # ============================================================

    if status != "APPROVED":

        st.divider()

        st.error(
            "🔒 FINAL REPORT LOCKED"
        )

        st.caption(
            "A PDF cannot be downloaded from this "
            "examination until a radiologist has "
            "reviewed and approved it."
        )

    # ============================================================
    # DISCLAIMER
    # ============================================================

    st.divider()

    st.caption(
        "Medusa AI provides AI-assisted screening "
        "information and does not replace professional "
        "medical diagnosis or treatment."
    )
