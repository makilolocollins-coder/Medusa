# ================================================================
# MEDUSA AI
# DETECTION + PATIENT REGISTRATION + MANDATORY RADIOLOGIST REVIEW
#
# IMPORTANT:
# Every scan MUST be reviewed and approved by a radiologist.
#
# AI analysis:
#     PENDING_REVIEW
#
# Radiologist approval:
#     RADIOLOGIST_APPROVED
#
# PDF download:
#     ONLY when RADIOLOGIST_APPROVED
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
    "Abia",
    "Adamawa",
    "Akwa Ibom",
    "Anambra",
    "Bauchi",
    "Bayelsa",
    "Benue",
    "Borno",
    "Cross River",
    "Delta",
    "Ebonyi",
    "Edo",
    "Ekiti",
    "Enugu",
    "Gombe",
    "Imo",
    "Jigawa",
    "Kaduna",
    "Kano",
    "Katsina",
    "Kebbi",
    "Kogi",
    "Kwara",
    "Lagos",
    "Nasarawa",
    "Niger",
    "Ogun",
    "Ondo",
    "Osun",
    "Oyo",
    "Plateau",
    "Rivers",
    "Sokoto",
    "Taraba",
    "Yobe",
    "Zamfara",
    "FCT",
]


# ================================================================
# CONSTANTS
# ================================================================

REVIEW_PENDING = "PENDING_REVIEW"
REVIEW_APPROVED = "RADIOLOGIST_APPROVED"


# ================================================================
# SESSION STATE
# ================================================================

def initialize_state():

    defaults = {
        "patient_id": None,
        "patient_name": "",
        "patient_state": "",
        "scan_id": None,
        "scan_model": None,
        "scan_type": None,
        "scan_result": None,
        "scan_image_bytes": None,
        "scan_filename": None,
        "review_status": None,
        "report_id": None,
        "report_pdf": None,
        "report_downloadable": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ================================================================
# PATIENT ID
# ================================================================

def generate_patient_id():

    date_part = datetime.now().strftime("%Y%m%d")

    unique_part = uuid.uuid4().hex[:8].upper()

    return f"MED-P-{date_part}-{unique_part}"


# ================================================================
# CURRENT USER
# ================================================================

def get_current_user():

    supabase = get_supabase()

    response = supabase.auth.get_user()

    if not response.user:
        return None

    return response.user


# ================================================================
# RESET EXAMINATION
# ================================================================

def reset_examination():

    keys = [
        "scan_id",
        "scan_model",
        "scan_type",
        "scan_result",
        "scan_image_bytes",
        "scan_filename",
        "review_status",
        "report_id",
        "report_pdf",
        "report_downloadable",
    ]

    for key in keys:
        st.session_state[key] = (
            False if key == "report_downloadable"
            else None
        )


# ================================================================
# LOAD REVIEW STATUS FROM DATABASE
# ================================================================

def get_review_status(scan_id):

    if not scan_id:
        return None

    try:

        supabase = get_supabase()

        response = (
            supabase
            .table("ai_scans")
            .select("status")
            .eq("id", scan_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            return None

        return response.data[0].get("status")

    except Exception:
        return None


# ================================================================
# MAIN
# ================================================================

def show_detection():

    initialize_state()

    set_background("detection.jpg")

    st.title("🧬 Medusa AI")

    st.caption(
        "AI-assisted medical image screening"
    )

    # ============================================================
    # PATIENT INFORMATION
    # ============================================================

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        patient_name = st.text_input(
            "Patient full name",
            value=st.session_state.patient_name,
            placeholder="Enter patient's full name",
            key="patient_name_input",
        )

    with col2:

        current_state = (
            st.session_state.patient_state
            if st.session_state.patient_state
            in NIGERIAN_STATES
            else None
        )

        state_index = (
            NIGERIAN_STATES.index(current_state)
            if current_state
            else 0
        )

        patient_state = st.selectbox(
            "State",
            NIGERIAN_STATES,
            index=state_index,
            key="patient_state_input",
        )

    patient_name = patient_name.strip()

    st.session_state.patient_name = patient_name
    st.session_state.patient_state = patient_state

    # ============================================================
    # PATIENT ID
    # ============================================================

    if (
        patient_name
        and st.session_state.patient_id is None
    ):

        st.session_state.patient_id = (
            generate_patient_id()
        )

    if st.session_state.patient_id:

        st.success(
            f"Patient ID: "
            f"**{st.session_state.patient_id}**"
        )

    # ============================================================
    # EXAMINATION
    # ============================================================

    st.subheader("Examination")

    model_choice = st.selectbox(
        "Select AI model",
        [
            "MammoSense — Breast Ultrasound",
            "MammoSense Pneumonia — Chest X-ray",
        ],
        key="detection_model",
    )

    is_pneumonia = (
        model_choice
        == "MammoSense Pneumonia — Chest X-ray"
    )

    if is_pneumonia:

        examination = "Chest X-ray"
        model_name = "MammoSense Pneumonia V2"

        uploader_label = "Upload chest X-ray"

        analyse_label = (
            "🔬 Analyze Chest X-ray"
        )

        st.info(
            "🫁 Upload a chest X-ray for "
            "AI-assisted pneumonia screening."
        )

    else:

        examination = "Breast Ultrasound"
        model_name = "MammoSense V2"

        uploader_label = (
            "Upload breast ultrasound"
        )

        analyse_label = (
            "🔬 Analyze Breast Ultrasound"
        )

        st.info(
            "🩻 Upload a breast ultrasound "
            "for AI-assisted screening."
        )

    # ============================================================
    # NEW EXAMINATION
    # ============================================================

    if st.session_state.scan_result is not None:

        if st.button(
            "＋ Start New Examination",
            use_container_width=True,
        ):

            reset_examination()

            st.rerun()

    # ============================================================
    # UPLOAD
    # ============================================================

    uploaded = st.file_uploader(
        uploader_label,
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="medical_image_upload",
    )

    if uploaded is not None:

        image_bytes = uploaded.getvalue()

        st.session_state.scan_image_bytes = (
            image_bytes
        )

        st.session_state.scan_filename = (
            uploaded.name
        )

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

        except Exception as error:

            st.error(
                "The uploaded image could not "
                "be opened."
            )

            st.exception(error)

            return

    elif st.session_state.scan_image_bytes:

        image_bytes = (
            st.session_state.scan_image_bytes
        )

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

        except Exception:

            st.error(
                "The stored scan image is invalid."
            )

            return

    else:

        st.warning(
            f"Upload a {examination.lower()} "
            "to continue."
        )

        return

    st.image(
        image,
        caption=examination,
        use_container_width=True,
    )

    # ============================================================
    # AI ANALYSIS
    # ============================================================

    if st.button(
        analyse_label,
        type="primary",
        use_container_width=True,
        key="analyse_selected_model",
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
                "Medusa AI is analyzing the image..."
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

            # ----------------------------------------------------
            # USER
            # ----------------------------------------------------

            user = get_current_user()

            if user is None:

                st.error(
                    "Your session has expired. "
                    "Please log in again."
                )

                return

            user_id = user.id

            supabase = get_supabase()

            # ----------------------------------------------------
            # IMAGE STORAGE
            # ----------------------------------------------------

            extension = (
                uploaded.name.split(".")[-1].lower()
                if uploaded is not None
                else "png"
            )

            image_path = (
                f"{user_id}/"
                f"{st.session_state.patient_id}/"
                f"{uuid.uuid4().hex}."
                f"{extension}"
            )

            content_type = (
                uploaded.type
                if uploaded is not None
                else "image/png"
            )

            supabase.storage.from_(
                "mammosense-scans"
            ).upload(
                image_path,
                image_bytes,
                {
                    "content-type": content_type,
                    "upsert": "false",
                },
            )

            # ----------------------------------------------------
            # SAVE SCAN
            #
            # CRITICAL:
            # Every new scan starts as PENDING_REVIEW.
            # ----------------------------------------------------

            scan_response = (
                supabase
                .table("ai_scans")
                .insert({
                    "user_id": user_id,

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
                        REVIEW_PENDING,
                })
                .execute()
            )

            if not scan_response.data:

                st.error(
                    "AI analysis completed, but "
                    "the scan could not be saved."
                )

                return

            scan_id = (
                scan_response.data[0]["id"]
            )

            # ----------------------------------------------------
            # SESSION
            # ----------------------------------------------------

            st.session_state.scan_result = result

            st.session_state.scan_id = scan_id

            st.session_state.scan_model = (
                model_name
            )

            st.session_state.scan_type = (
                "pneumonia"
                if is_pneumonia
                else "mammosense"
            )

            st.session_state.review_status = (
                REVIEW_PENDING
            )

            st.session_state.report_pdf = None

            st.session_state.report_id = None

            st.session_state.report_downloadable = (
                False
            )

            st.success(
                "✅ AI analysis completed."
            )

            st.warning(
                "🔒 Radiologist review is mandatory. "
                "The final report is currently locked."
            )

        except Exception as error:

            st.error(
                "The AI model could not analyze "
                "this image."
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

    current_model = (
        st.session_state.scan_model
        or model_name
    )

    st.caption(
        f"Model: {current_model}"
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

    if (
        isinstance(probabilities, dict)
        and probabilities
    ):

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
    # REFRESH STATUS FROM SUPABASE
    # ============================================================

    database_status = get_review_status(
        st.session_state.scan_id
    )

    if database_status:

        st.session_state.review_status = (
            database_status
        )

    status = st.session_state.review_status

    # ============================================================
    # REVIEW STATUS
    # ============================================================

    st.divider()

    st.subheader(
        "👨‍⚕️ Radiologist Review"
    )

    if status == REVIEW_APPROVED:

        st.success(
            "✅ This examination has been "
            "reviewed and approved by a radiologist."
        )

    elif status == REVIEW_PENDING:

        st.warning(
            "⏳ Radiologist review required."
        )

        st.info(
            "This AI result is preliminary. "
            "A qualified radiologist must review "
            "and approve this examination before "
            "a final medical report can be downloaded."
        )

    else:

        st.warning(
            "🔒 This examination has not yet "
            "received radiologist approval."
        )

    # ============================================================
    # DOWNLOAD
    #
    # HARD LOCK
    #
    # We deliberately DO NOT generate a PDF here.
    # ============================================================

    st.divider()

    st.subheader(
        "📄 Final Medical Report"
    )

    if status == REVIEW_APPROVED:

        st.success(
            "✓ Radiologist-approved report available."
        )

        # --------------------------------------------------------
        # Download only if the approved report exists.
        # The actual PDF generation/storage is performed
        # by the radiologist review workflow.
        # --------------------------------------------------------

        if st.session_state.report_pdf:

            st.download_button(
                label=(
                    "⬇️ Download Final Medical Report"
                ),
                data=st.session_state.report_pdf,
                file_name=(
                    f"{st.session_state.report_id}.pdf"
                ),
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="download_final_report",
            )

        else:

            st.info(
                "The examination is approved, "
                "but the final report is not loaded "
                "in this session."
            )

    else:

        st.error(
            "🔒 DOWNLOAD LOCKED"
        )

        st.caption(
            "No final PDF can be downloaded. "
            "Radiologist review and approval are "
            "mandatory for every examination."
        )

    # ============================================================
    # DISCLAIMER
    # ============================================================

    st.divider()

    st.caption(
        "Medusa AI provides AI-assisted screening "
        "information and does not replace professional "
        "medical evaluation, diagnosis, or treatment."
    )
