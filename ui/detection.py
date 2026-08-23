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


# ============================================================
# NIGERIAN STATES
# ============================================================

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


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():

    defaults = {
        "medusa_patient_id": None,
        "medusa_patient_name": "",
        "medusa_patient_state": "Delta",

        "medusa_scan_id": None,
        "medusa_scan_result": None,
        "medusa_scan_model": None,
        "medusa_scan_type": None,
        "medusa_scan_image": None,
        "medusa_scan_filename": None,

        "medusa_review_status": "NOT_REQUESTED",
        "medusa_review_id": None,

        "medusa_report_id": None,
        "medusa_report_path": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# PATIENT ID
# ============================================================

def generate_patient_id():

    return (
        "MED-P-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:8].upper()
    )


# ============================================================
# CURRENT USER
# ============================================================

def get_current_user():

    try:

        supabase = get_supabase()
        response = supabase.auth.get_user()

        if response.user:
            return response.user

    except Exception:
        return None

    return None


# ============================================================
# RESET EXAMINATION
# ============================================================

def reset_examination():

    st.session_state.medusa_scan_id = None
    st.session_state.medusa_scan_result = None
    st.session_state.medusa_scan_model = None
    st.session_state.medusa_scan_type = None
    st.session_state.medusa_scan_image = None
    st.session_state.medusa_scan_filename = None

    st.session_state.medusa_review_status = "NOT_REQUESTED"
    st.session_state.medusa_review_id = None

    st.session_state.medusa_report_id = None
    st.session_state.medusa_report_path = None


# ============================================================
# GET REVIEW STATUS
# ============================================================

def get_review_status(scan_id):

    if not scan_id:
        return "NOT_REQUESTED", None

    try:

        supabase = get_supabase()

        reviews = (
            supabase
            .table("radiologist_reviews")
            .select("id,status,approved")
            .eq("scan_id", scan_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )

        if reviews:

            review = reviews[0]

            status = str(
                review.get("status", "")
            ).upper()

            approved = review.get(
                "approved",
                False,
            )

            if status == "APPROVED" or approved is True:

                return "APPROVED", review.get("id")

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

        if requests:

            request = requests[0]

            request_status = str(
                request.get("status", "")
            ).upper()

            if request_status in (
                "PENDING",
                "REQUESTED",
            ):

                return "PENDING", None

            if request_status in (
                "APPROVED",
                "COMPLETED",
                "REVIEWED",
            ):

                return "APPROVED", None

        return "NOT_REQUESTED", None

    except Exception:
        return (
            st.session_state.medusa_review_status,
            st.session_state.medusa_review_id,
        )


# ============================================================
# GET APPROVED REPORT
# ============================================================

def get_approved_report(scan_id):

    if not scan_id:
        return None

    try:

        supabase = get_supabase()

        reports = (
            supabase
            .table("medical_reports")
            .select(
                "report_id,pdf_path,status,approved_at"
            )
            .eq(
                "scan_id",
                scan_id,
            )
            .eq(
                "status",
                "APPROVED",
            )
            .order(
                "approved_at",
                desc=True,
            )
            .limit(1)
            .execute()
            .data
            or []
        )

        if reports:
            return reports[0]

    except Exception:
        pass

    return None


# ============================================================
# SHOW DETECTION
# ============================================================

def show_detection():

    initialize_state()

    set_background("detection.jpg")

    st.title("Medusa AI")

    st.caption(
        "AI-assisted medical imaging and radiologist review"
    )

    # ========================================================
    # PATIENT REGISTRATION
    # ========================================================

    st.subheader("Patient Registration")

    patient_name = st.text_input(
        "Patient full name",
        value=st.session_state.medusa_patient_name,
        placeholder="Enter patient's full name",
        key="medusa_patient_name_widget",
    )

    state_index = 0

    if (
        st.session_state.medusa_patient_state
        in NIGERIAN_STATES
    ):

        state_index = NIGERIAN_STATES.index(
            st.session_state.medusa_patient_state
        )

    patient_state = st.selectbox(
        "State",
        NIGERIAN_STATES,
        index=state_index,
        key="medusa_patient_state_widget",
    )

    patient_name = patient_name.strip()

    # Store values in our OWN session keys.
    # The widget keys are deliberately different.

    st.session_state.medusa_patient_name = patient_name
    st.session_state.medusa_patient_state = patient_state

    if patient_name and not st.session_state.medusa_patient_id:

        st.session_state.medusa_patient_id = (
            generate_patient_id()
        )

    if st.session_state.medusa_patient_id:

        st.info(
            f"Patient ID: "
            f"{st.session_state.medusa_patient_id}"
        )

    # ========================================================
    # NEW EXAMINATION
    # ========================================================

    if st.session_state.medusa_scan_result is not None:

        if st.button(
            "Start New Examination",
            use_container_width=True,
            key="medusa_new_examination",
        ):

            reset_examination()

            st.rerun()

    # ========================================================
    # EXAMINATION TYPE
    # ========================================================

    st.subheader("Examination")

    model_choice = st.selectbox(
        "Select AI model",
        [
            "MammoSense — Breast Ultrasound",
            "MammoSense Pneumonia — Chest X-ray",
        ],
        key="medusa_model_choice",
    )

    is_pneumonia = (
        model_choice
        == "MammoSense Pneumonia — Chest X-ray"
    )

    if is_pneumonia:

        examination = "Chest X-ray"
        model_name = "MammoSense Pneumonia V2"

        st.info(
            "Upload a chest X-ray for AI-assisted "
            "pneumonia screening."
        )

    else:

        examination = "Breast Ultrasound"
        model_name = "MammoSense V2"

        st.info(
            "Upload a breast ultrasound for "
            "AI-assisted screening."
        )

    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    uploaded = st.file_uploader(
        "Upload medical image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="medusa_medical_upload",
    )

    if uploaded is not None:

        image_bytes = uploaded.getvalue()

        st.session_state.medusa_scan_image = image_bytes
        st.session_state.medusa_scan_filename = uploaded.name

    else:

        image_bytes = (
            st.session_state.medusa_scan_image
        )

    if not image_bytes:

        st.warning(
            "Upload an image to continue."
        )

        return

    # ========================================================
    # OPEN IMAGE
    # ========================================================

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        st.error(
            "The uploaded image could not be opened."
        )

        return

    st.image(
        image,
        caption=examination,
        use_container_width=True,
    )

    # ========================================================
    # ANALYZE
    # ========================================================

    if st.button(
        "Analyze Examination",
        type="primary",
        use_container_width=True,
        key="medusa_analyze_button",
    ):

        if not patient_name:

            st.error(
                "Enter the patient's full name first."
            )

            return

        user = get_current_user()

        if user is None:

            st.error(
                "Your login session has expired. "
                "Please log in again."
            )

            return

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

            supabase = get_supabase()

            # ------------------------------------------------
            # SAVE IMAGE
            # ------------------------------------------------

            extension = "png"

            content_type = "image/png"

            if uploaded is not None:

                if "." in uploaded.name:

                    extension = (
                        uploaded.name
                        .rsplit(".", 1)[-1]
                        .lower()
                    )

                content_type = (
                    uploaded.type
                    or "image/png"
                )

            image_path = (
                f"{user.id}/"
                f"{st.session_state.medusa_patient_id}/"
                f"{uuid.uuid4().hex}."
                f"{extension}"
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

            # ------------------------------------------------
            # SAVE SCAN
            # ------------------------------------------------

            scan_response = (
                supabase
                .table("ai_scans")
                .insert({
                    "user_id": user.id,
                    "patient_id": (
                        st.session_state.medusa_patient_id
                    ),
                    "patient_name": patient_name,
                    "patient_state": patient_state,
                    "examination": examination,
                    "model": model_name,
                    "prediction": result.get(
                        "prediction",
                        "Unknown",
                    ),
                    "confidence": float(
                        result.get(
                            "confidence",
                            0,
                        )
                    ),
                    "probabilities": result.get(
                        "probabilities",
                        {},
                    ),
                    "image_path": image_path,
                    "status": "AI_COMPLETED",
                })
                .execute()
            )

            if not scan_response.data:

                st.error(
                    "AI analysis completed, but "
                    "the scan could not be saved."
                )

                return

            scan_id = scan_response.data[0]["id"]

            # ------------------------------------------------
            # SESSION
            # ------------------------------------------------

            st.session_state.medusa_scan_id = scan_id

            st.session_state.medusa_scan_result = result

            st.session_state.medusa_scan_model = model_name

            st.session_state.medusa_scan_type = (
                "pneumonia"
                if is_pneumonia
                else "mammosense"
            )

            st.session_state.medusa_review_status = (
                "NOT_REQUESTED"
            )

            st.session_state.medusa_review_id = None

            st.session_state.medusa_report_id = None

            st.session_state.medusa_report_path = None

            st.success(
                "AI analysis completed successfully."
            )

            st.rerun()

        except Exception as error:

            st.error(
                "The examination could not be analyzed."
            )

            st.exception(error)

            return

    # ========================================================
    # RESULT
    # ========================================================

    result = st.session_state.medusa_scan_result

    if result is None:
        return

    # ========================================================
    # REVIEW STATUS
    # ========================================================

    status, review_id = get_review_status(
        st.session_state.medusa_scan_id
    )

    st.session_state.medusa_review_status = status

    if review_id:

        st.session_state.medusa_review_id = review_id

    # ========================================================
    # AI RESULT
    # ========================================================

    st.divider()

    st.subheader("AI Screening Result")

    prediction = str(
        result.get(
            "prediction",
            "Unknown",
        )
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
            "MALIGNANT",
            "PNEUMONIA",
        ):

            st.error(
                f"Finding: {prediction}"
            )

        else:

            st.success(
                f"Finding: {prediction}"
            )

    with col2:

        st.metric(
            "AI Confidence",
            f"{confidence:.1%}",
        )

    # ========================================================
    # PROBABILITIES
    # ========================================================

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

        for label, value in probabilities.items():

            try:

                value = float(value)

            except Exception:

                continue

            st.write(
                f"**{label}: {value:.2%}**"
            )

            st.progress(
                min(
                    max(
                        value,
                        0.0,
                    ),
                    1.0,
                )
            )

    # ========================================================
    # RADIOLOGIST WORKFLOW
    # ========================================================

    st.divider()

    st.subheader(
        "Radiologist Review"
    )

    st.info(
        "Every examination requires review and "
        "approval by a qualified radiologist "
        "before the final medical report can "
        "be downloaded."
    )

    # --------------------------------------------------------
    # NOT REQUESTED
    # --------------------------------------------------------

    if status == "NOT_REQUESTED":

        st.warning(
            "This examination has not yet been "
            "submitted for radiologist review."
        )

        if st.button(
            "Submit Examination for Radiologist Review",
            type="primary",
            use_container_width=True,
            key="medusa_submit_review",
        ):

            user = get_current_user()

            if user is None:

                st.error(
                    "Please log in again."
                )

                return

            try:

                supabase = get_supabase()

                # Check whether request already exists.

                existing = (
                    supabase
                    .table("radiologist_requests")
                    .select("id,status")
                    .eq(
                        "scan_id",
                        st.session_state.medusa_scan_id,
                    )
                    .order(
                        "created_at",
                        desc=True,
                    )
                    .limit(1)
                    .execute()
                    .data
                    or []
                )

                if existing:

                    st.session_state.medusa_review_status = (
                        "PENDING"
                    )

                    st.info(
                        "This examination is already "
                        "in the radiologist review queue."
                    )

                    st.rerun()

                request_response = (
                    supabase
                    .table("radiologist_requests")
                    .insert({
                        "user_id": user.id,
                        "scan_id": (
                            st.session_state.medusa_scan_id
                        ),
                        "status": "PENDING",
                    })
                    .execute()
                )

                if not request_response.data:

                    st.error(
                        "The review request could "
                        "not be created."
                    )

                    return

                # Update scan status.

                (
                    supabase
                    .table("ai_scans")
                    .update({
                        "status":
                            "AWAITING_RADIOLOGIST",
                    })
                    .eq(
                        "id",
                        st.session_state.medusa_scan_id,
                    )
                    .execute()
                )

                st.session_state.medusa_review_status = (
                    "PENDING"
                )

                st.success(
                    "Examination submitted to the "
                    "radiologist successfully."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "Could not submit the examination "
                    "for radiologist review."
                )

                st.exception(error)

    # --------------------------------------------------------
    # PENDING
    # --------------------------------------------------------

    elif status == "PENDING":

        st.warning(
            "⏳ Radiologist review is pending."
        )

        st.caption(
            "The radiologist must review and approve "
            "this examination before a report can "
            "be downloaded."
        )

    # --------------------------------------------------------
    # APPROVED
    # --------------------------------------------------------

    elif status == "APPROVED":

        st.success(
            "✓ Radiologist review completed and approved."
        )

        # ----------------------------------------------------
        # GET REPORT
        # ----------------------------------------------------

        report = get_approved_report(
            st.session_state.medusa_scan_id
        )

        if report:

            report_id = report.get(
                "report_id"
            )

            pdf_path = report.get(
                "pdf_path"
            )

            st.session_state.medusa_report_id = (
                report_id
            )

            st.session_state.medusa_report_path = (
                pdf_path
            )

            if pdf_path:

                try:

                    supabase = get_supabase()

                    pdf_bytes = (
                        supabase
                        .storage
                        .from_(
                            "medical-reports"
                        )
                        .download(
                            pdf_path
                        )
                    )

                    st.success(
                        f"Final report approved. "
                        f"Report ID: {report_id}"
                    )

                    st.download_button(
                        "Download Final Medical Report",
                        data=pdf_bytes,
                        file_name=(
                            f"{report_id}.pdf"
                        ),
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                        key=(
                            "medusa_download_"
                            + str(report_id)
                        ),
                    )

                except Exception as error:

                    st.error(
                        "The report is approved, "
                        "but the PDF could not be downloaded."
                    )

                    st.exception(error)

            else:

                st.warning(
                    "The examination is approved, "
                    "but the report file is not available."
                )

        else:

            st.info(
                "The radiologist has approved the "
                "examination. The final report is "
                "being prepared."
            )

    # ========================================================
    # SECURITY NOTICE
    # ========================================================

    st.divider()

    if status != "APPROVED":

        st.error(
            "DOWNLOAD LOCKED"
        )

        st.caption(
            "A final medical report cannot be downloaded "
            "until radiologist review and approval "
            "are recorded."
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "Medusa AI provides AI-assisted screening "
        "information and does not replace professional "
        "medical diagnosis or treatment."
    )
