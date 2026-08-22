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

from reports.pdf_report import generate_pdf_report


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
        "review_status": "NOT_REQUESTED",
        "review_id": None,
        "report_pdf": None,
        "report_id": None,
        "report_downloadable": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


def generate_patient_id():

    return (
        "MED-P-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:8].upper()
    )


def get_current_user():

    try:

        supabase = get_supabase()
        response = supabase.auth.get_user()

        if response.user:
            return response.user

    except Exception:
        pass

    return None


def reset_examination():

    keys = [
        "scan_id",
        "scan_model",
        "scan_type",
        "scan_result",
        "scan_image_bytes",
        "scan_filename",
        "review_id",
        "report_pdf",
        "report_id",
    ]

    for key in keys:
        st.session_state[key] = None

    st.session_state.review_status = "NOT_REQUESTED"
    st.session_state.report_downloadable = False


def load_latest_review_status():

    scan_id = st.session_state.get("scan_id")

    if not scan_id:
        return

    try:

        supabase = get_supabase()

        requests = (
            supabase
            .table("radiologist_requests")
            .select("*")
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
                request.get("status", "PENDING")
            ).upper()

            if request_status in (
                "APPROVED",
                "COMPLETED",
                "REVIEWED",
            ):

                st.session_state.review_status = "APPROVED"

            elif request_status in (
                "PENDING",
                "REQUESTED",
            ):

                st.session_state.review_status = "PENDING"

            else:

                st.session_state.review_status = (
                    request_status
                )

        reviews = (
            supabase
            .table("radiologist_reviews")
            .select("*")
            .eq("scan_id", scan_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )

        if reviews:

            review = reviews[0]

            st.session_state.review_id = review.get("id")

            review_status = str(
                review.get("status", "")
            ).upper()

            if (
                review_status == "APPROVED"
                or review.get("approved") is True
            ):

                st.session_state.review_status = "APPROVED"

    except Exception:
        pass


def show_detection():

    initialize_state()

    set_background("detection.jpg")

    st.title("Medusa AI")

    st.caption(
        "AI-assisted medical imaging and radiologist review"
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
            placeholder="Enter patient's full name",
            key="patient_name",
        )

    with col2:

        patient_state = st.selectbox(
            "State",
            NIGERIAN_STATES,
            index=(
                NIGERIAN_STATES.index(
                    st.session_state.patient_state
                )
                if st.session_state.patient_state
                in NIGERIAN_STATES
                else 0
            ),
            key="patient_state",
        )

    patient_name = patient_name.strip()

    if patient_name:

        st.session_state.patient_name = patient_name

        if not st.session_state.patient_id:

            st.session_state.patient_id = (
                generate_patient_id()
            )

    st.session_state.patient_state = patient_state

    if st.session_state.patient_id:

        st.info(
            f"Patient ID: {st.session_state.patient_id}"
        )

    # ============================================================
    # NEW EXAMINATION
    # ============================================================

    if st.session_state.scan_result is not None:

        if st.button(
            "Start New Examination",
            use_container_width=True,
        ):

            reset_examination()
            st.rerun()

    # ============================================================
    # MODEL
    # ============================================================

    st.subheader("Examination")

    model_choice = st.selectbox(
        "AI model",
        [
            "MammoSense — Breast Ultrasound",
            "MammoSense Pneumonia — Chest X-ray",
        ],
        key="medical_model",
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

    # ============================================================
    # IMAGE
    # ============================================================

    uploaded = st.file_uploader(
        "Upload medical image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="medical_scan_upload",
    )

    if uploaded is not None:

        image_bytes = uploaded.getvalue()

        st.session_state.scan_image_bytes = image_bytes
        st.session_state.scan_filename = uploaded.name

    else:

        image_bytes = (
            st.session_state.scan_image_bytes
        )

    if not image_bytes:

        st.warning(
            "Upload an image to continue."
        )

        return

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
        "Analyze Examination",
        type="primary",
        use_container_width=True,
        key="analyze_medical_scan",
    ):

        if not patient_name:

            st.error(
                "Enter the patient's full name first."
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

            user = get_current_user()

            if user is None:

                st.error(
                    "Your login session has expired."
                )

                return

            supabase = get_supabase()

            extension = (
                uploaded.name.split(".")[-1].lower()
                if uploaded
                else "png"
            )

            image_path = (
                f"{user.id}/"
                f"{st.session_state.patient_id}/"
                f"{uuid.uuid4().hex}.{extension}"
            )

            content_type = (
                uploaded.type
                if uploaded
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

            response = (
                supabase
                .table("ai_scans")
                .insert({
                    "user_id": user.id,
                    "patient_id": (
                        st.session_state.patient_id
                    ),
                    "patient_name": patient_name,
                    "patient_state": patient_state,
                    "examination": examination,
                    "model": model_name,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": result.get(
                        "probabilities",
                        {},
                    ),
                    "image_path": image_path,
                    "status": "AI_COMPLETED",
                })
                .execute()
            )

            if not response.data:

                st.error(
                    "The scan could not be saved."
                )

                return

            scan_id = response.data[0]["id"]

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
            st.session_state.report_pdf = None
            st.session_state.report_id = None
            st.session_state.report_downloadable = False

            st.success(
                "AI analysis completed."
            )

        except Exception as error:

            st.error(
                "The examination could not be analyzed."
            )

            st.exception(error)

            return

    # ============================================================
    # RESULT
    # ============================================================

    result = st.session_state.scan_result

    if result is None:
        return

    load_latest_review_status()

    st.divider()

    st.subheader("AI Screening Result")

    prediction = result.get(
        "prediction",
        "Unknown",
    )

    confidence = float(
        result.get("confidence", 0)
    )

    col1, col2 = st.columns(2)

    with col1:

        if prediction.upper() in (
            "PNEUMONIA",
            "MALIGNANT",
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

    probabilities = result.get(
        "probabilities",
        {},
    )

    if isinstance(probabilities, dict):

        st.subheader(
            "Probability Breakdown"
        )

        for label, value in probabilities.items():

            value = float(value)

            st.write(
                f"{label}: {value:.2%}"
            )

            st.progress(
                min(max(value, 0), 1)
            )

    # ============================================================
    # RADIOLOGIST WORKFLOW
    # ============================================================

    st.divider()

    st.subheader(
        "Radiologist Review"
    )

    st.info(
        "Every examination must be reviewed by "
        "a qualified radiologist before a final "
        "medical report can be downloaded."
    )

    status = st.session_state.review_status

    if status == "APPROVED":

        st.success(
            "Radiologist review completed and approved."
        )

    elif status == "PENDING":

        st.warning(
            "Radiologist review is pending."
        )

    else:

        st.warning(
            "This examination has not yet been "
            "submitted for radiologist review."
        )

        if st.button(
            "Submit Examination for Radiologist Review",
            type="primary",
            use_container_width=True,
            key="submit_radiologist_request",
        ):

            user = get_current_user()

            if user is None:

                st.error(
                    "Please log in again."
                )

                return

            try:

                supabase = get_supabase()

                # IMPORTANT:
                # This inserts ONLY into radiologist_requests.
                #
                # We DO NOT insert into radiologist_reviews.
                #
                # This fixes:
                # null value in column radiologist_name

                existing = (
                    supabase
                    .table("radiologist_requests")
                    .select("id,status")
                    .eq(
                        "scan_id",
                        st.session_state.scan_id,
                    )
                    .limit(1)
                    .execute()
                    .data
                    or []
                )

                if existing:

                    st.session_state.review_status = (
                        "PENDING"
                    )

                    st.info(
                        "This examination has already "
                        "been submitted for review."
                    )

                else:

                    request_response = (
                        supabase
                        .table(
                            "radiologist_requests"
                        )
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
                            "not be created."
                        )

                        return

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

                    st.session_state.review_status = (
                        "PENDING"
                    )

                    st.success(
                        "Examination submitted successfully "
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
    # FINAL REPORT
    # ============================================================

    st.divider()

    st.subheader(
        "Final Medical Report"
    )

    # ============================================================
    # IMPORTANT:
    # PATIENT CANNOT CREATE OR EDIT REVIEW.
    # PATIENT CANNOT GENERATE REPORT.
    # PATIENT CANNOT DOWNLOAD REPORT UNTIL
    # RADIOLOGIST APPROVAL EXISTS.
    # ============================================================

    if status == "APPROVED":

        st.success(
            "Final radiologist-approved report is available."
        )

        # --------------------------------------------------------
        # Retrieve approved report
        # --------------------------------------------------------

        try:

            supabase = get_supabase()

            reports = (
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
                .order(
                    "approved_at",
                    desc=True,
                )
                .limit(1)
                .execute()
                .data
                or []
            )

        except Exception:

            reports = []

        if reports:

            report = reports[0]

            report_id = report.get(
                "report_id"
            )

            pdf_path = report.get(
                "pdf_path"
            )

            if pdf_path:

                try:

                    pdf_response = (
                        supabase
                        .storage
                        .from_("medical-reports")
                        .download(pdf_path)
                    )

                    st.download_button(
                        "Download Final Medical Report",
                        data=pdf_response,
                        file_name=(
                            f"{report_id}.pdf"
                        ),
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                        key="download_approved_report",
                    )

                except Exception as error:

                    st.error(
                        "The report was approved but "
                        "could not be downloaded."
                    )

                    st.exception(error)

        else:

            st.info(
                "The radiologist has approved the examination. "
                "The final report is being prepared."
            )

    else:

        st.error(
            "DOWNLOAD LOCKED"
        )

        st.caption(
            "A final medical report cannot be downloaded "
            "until a radiologist has reviewed and approved "
            "this examination."
        )

    # ============================================================
    # DISCLAIMER
    # ============================================================

    st.divider()

    st.caption(
        "Medusa AI provides AI-assisted screening "
        "and does not replace professional medical "
        "diagnosis or treatment."
    )
