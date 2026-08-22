# ================================================================
# MEDUSA AI
# PATIENT DETECTION
#
# PATIENT WORKFLOW:
# Patient information
#       ↓
# AI analysis
#       ↓
# Request radiologist review
#       ↓
# LOCKED
#
# IMPORTANT:
# This file NEVER creates a radiologist_reviews record.
# Radiologist review is performed ONLY in radiologist.py.
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
        "scan_image_path": None,
        "review_id": None,
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

    if response.user:
        return response.user

    return None


# ================================================================
# RESET
# ================================================================

def reset_examination():

    keys = {
        "scan_id": None,
        "scan_model": None,
        "scan_type": None,
        "scan_result": None,
        "scan_image_bytes": None,
        "scan_filename": None,
        "scan_image_path": None,
        "review_id": None,
        "review_status": None,
        "report_id": None,
        "report_pdf": None,
        "report_downloadable": False,
    }

    for key, value in keys.items():
        st.session_state[key] = value


# ================================================================
# CHECK REVIEW STATUS
#
# IMPORTANT:
# Patient only READS the review status.
# Patient NEVER CREATES radiologist_reviews.
# ================================================================

def get_review_status(scan_id):

    if not scan_id:
        return None, None

    try:

        supabase = get_supabase()

        requests = (
            supabase
            .table("radiologist_requests")
            .select("id,status")
            .eq("scan_id", scan_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )

        if not requests:
            return None, None

        request = requests[0]

        request_id = request.get("id")

        request_status = (
            request.get("status")
            or "PENDING"
        )

        # --------------------------------------------------------
        # Check whether radiologist has completed the review.
        # --------------------------------------------------------

        reviews = (
            supabase
            .table("radiologist_reviews")
            .select(
                "id,status,approved,reviewed_at"
            )
            .eq("scan_id", scan_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )

        if reviews:

            review = reviews[0]

            if (
                review.get("approved") is True
                or str(
                    review.get("status", "")
                ).upper()
                == "APPROVED"
            ):

                return request_id, "APPROVED"

        return request_id, str(
            request_status
        ).upper()

    except Exception:
        return None, None


# ================================================================
# LOAD EXISTING APPROVED REPORT
# ================================================================

def load_existing_report(scan_id):

    if not scan_id:
        return None

    try:

        supabase = get_supabase()

        reports = (
            supabase
            .table("medical_reports")
            .select(
                "report_id,status,pdf_path"
            )
            .eq("scan_id", scan_id)
            .eq("status", "APPROVED")
            .order("approved_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )

        if not reports:
            return None

        return reports[0]

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
        "AI-assisted medical imaging screening"
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

        previous_state = (
            st.session_state.patient_state
            if st.session_state.patient_state
            in NIGERIAN_STATES
            else None
        )

        state_index = (
            NIGERIAN_STATES.index(
                previous_state
            )
            if previous_state
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

        st.info(
            f"Patient ID: "
            f"{st.session_state.patient_id}"
        )

    # ============================================================
    # EXAMINATION
    # ============================================================

    st.subheader("Examination")

    model_choice = st.selectbox(
        "Select AI examination",
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

        model_name = (
            "MammoSense Pneumonia V2"
        )

        uploader_label = (
            "Upload chest X-ray"
        )

        analyse_label = (
            "🔬 Analyze Chest X-ray"
        )

        st.info(
            "🫁 AI-assisted pneumonia screening."
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
            "🩻 AI-assisted breast ultrasound screening."
        )

    # ============================================================
    # NEW EXAMINATION
    # ============================================================

    if st.session_state.scan_result is not None:

        if st.button(
            "＋ Start New Examination",
            use_container_width=True,
            key="new_examination",
        ):

            reset_examination()

            st.rerun()

    # ============================================================
    # IMAGE UPLOAD
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

        st.image(
            image,
            caption=examination,
            use_container_width=True,
        )

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
                "Stored examination image "
                "could not be opened."
            )

            return

        st.image(
            image,
            caption=examination,
            use_container_width=True,
        )

    else:

        st.warning(
            f"Upload a {examination.lower()} "
            "to continue."
        )

        return

    # ============================================================
    # ANALYSIS
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
                "Medusa AI is analyzing the examination..."
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
            # SAVE AI SCAN
            # ----------------------------------------------------

            scan_data = {
                "user_id": user.id,
                "patient_id": (
                    st.session_state.patient_id
                ),
                "patient_name": patient_name,
                "patient_state": patient_state,
                "examination": examination,
                "model": model_name,
                "prediction": result.get(
                    "prediction"
                ),
                "confidence": result.get(
                    "confidence"
                ),
                "probabilities": result.get(
                    "probabilities",
                    {},
                ),
                "image_path": image_path,
                "status": "AI_COMPLETED",
            }

            scan_response = (
                supabase
                .table("ai_scans")
                .insert(scan_data)
                .execute()
            )

            if not scan_response.data:

                st.error(
                    "AI analysis completed, but "
                    "the scan could not be saved."
                )

                return

            scan_id = scan_response.data[0]["id"]

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

            st.session_state.scan_image_path = (
                image_path
            )

            st.session_state.review_id = None

            st.session_state.review_status = (
                None
            )

            st.session_state.report_pdf = None

            st.session_state.report_id = None

            st.session_state.report_downloadable = (
                False
            )

            st.success(
                "✅ AI analysis completed."
            )

            st.rerun()

        except Exception as error:

            st.error(
                "The AI examination could not be completed."
            )

            st.exception(error)

            return

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

    st.caption(
        f"Model: "
        f"{st.session_state.scan_model}"
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

    probabilities = result.get(
        "probabilities",
        {},
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
    # RADIOLOGIST REVIEW STATUS
    # ============================================================

    st.divider()

    st.subheader(
        "👨‍⚕️ Radiologist Review"
    )

    st.info(
        "Every examination must be reviewed by "
        "a qualified radiologist before a final "
        "medical report becomes available."
    )

    request_id, status = get_review_status(
        st.session_state.scan_id
    )

    if status:

        st.session_state.review_id = request_id

        st.session_state.review_status = status

    else:

        status = (
            st.session_state.review_status
        )

    # ============================================================
    # NO REVIEW REQUEST
    # ============================================================

    if not status:

        st.warning(
            "⏳ This examination has not yet "
            "been submitted for radiologist review."
        )

        if st.button(
            "📋 Request Radiologist Review",
            type="primary",
            use_container_width=True,
            key="request_radiologist_review",
        ):

            try:

                user = get_current_user()

                if user is None:

                    st.error(
                        "Please log in again."
                    )

                    return

                supabase = get_supabase()

                # ------------------------------------------------
                # IMPORTANT
                #
                # ONLY radiologist_requests is inserted here.
                #
                # NO radiologist_name
                # NO findings
                # NO impression
                # NO recommendations
                # NO remarks
                #
                # Those belong to radiologist.py.
                # ------------------------------------------------

                request_response = (
                    supabase
                    .table("radiologist_requests")
                    .insert({
                        "user_id": user.id,
                        "scan_id": (
                            st.session_state.scan_id
                        ),
                        "status": "PENDING",
                    })
                    .execute()
                )

                if not request_response.data:

                    st.error(
                        "The review request could "
                        "not be submitted."
                    )

                    return

                request = (
                    request_response.data[0]
                )

                st.session_state.review_id = (
                    request.get("id")
                )

                st.session_state.review_status = (
                    "PENDING"
                )

                # Update scan status

                (
                    supabase
                    .table("ai_scans")
                    .update({
                        "status":
                            "AWAITING_RADIOLOGIST",
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

        st.info(
            "⏳ Radiologist review is pending."
        )

        st.caption(
            "The final report remains locked "
            "until the radiologist completes "
            "and approves the review."
        )

        st.error(
            "🔒 FINAL PDF LOCKED"
        )

    # ============================================================
    # APPROVED
    # ============================================================

    elif status == "APPROVED":

        st.success(
            "✅ Radiologist review completed "
            "and approved."
        )

        # --------------------------------------------------------
        # LOAD APPROVED REPORT
        # --------------------------------------------------------

        report = load_existing_report(
            st.session_state.scan_id
        )

        if report:

            report_id = report.get(
                "report_id"
            )

            st.session_state.report_id = (
                report_id
            )

            st.session_state.report_downloadable = (
                True
            )

            st.success(
                f"Final report available\n\n"
                f"Report ID: **{report_id}**"
            )

            # ----------------------------------------------------
            # Download PDF from storage
            # ----------------------------------------------------

            if st.button(
                "⬇️ Prepare Final Medical Report",
                use_container_width=True,
                key="prepare_report",
            ):

                try:

                    supabase = get_supabase()

                    pdf_path = report.get(
                        "pdf_path"
                    )

                    pdf_bytes = (
                        supabase.storage
                        .from_("medical-reports")
                        .download(pdf_path)
                    )

                    st.session_state.report_pdf = (
                        pdf_bytes
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "Could not retrieve the "
                        "approved report."
                    )

                    st.exception(error)

            if st.session_state.report_pdf:

                st.download_button(
                    label=(
                        "⬇️ Download Final "
                        "Medical Report"
                    ),
                    data=(
                        st.session_state.report_pdf
                    ),
                    file_name=(
                        f"{report_id}.pdf"
                    ),
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key="download_final_report",
                )

        else:

            st.warning(
                "The radiologist has approved "
                "the examination, but the final "
                "report is still being prepared."
            )

    # ============================================================
    # UNKNOWN STATUS
    # ============================================================

    else:

        st.warning(
            f"Review status: {status}"
        )

        st.error(
            "🔒 FINAL PDF LOCKED"
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
