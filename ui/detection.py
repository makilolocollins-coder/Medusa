# ============================================================
# MEDUSA AI
# AI MEDICAL DETECTION PAGE
#
# Supports:
#   1. MammoSense Breast Ultrasound
#   2. MammoSense Pneumonia
#   3. MammoSense Tuberculosis
#   4. MammoSense Brain MRI
#
# Brain MRI:
#   2D ResNet-50 image classifier
#
# Classes:
#   glioma
#   meningioma
#   pituitary
#   notumor
# ============================================================

import io
import uuid
from datetime import datetime

import streamlit as st
from PIL import Image

from ui.background import set_background
from utils.supabase_client import get_supabase


# ============================================================
# NIGERIAN STATES
# ============================================================

STATES = [
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

def init_state():

    defaults = {
        "patient_id": None,
        "scan_id": None,
        "scan_result": None,
        "scan_image_bytes": None,
        "scan_filename": None,
        "review_status": "NOT_REQUESTED",
        "review_id": None,
        "patient_name_input": "",
        "patient_state_input": "Delta",
        "medical_model_input": (
            "MammoSense — Breast Ultrasound"
        ),
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


# ============================================================
# CURRENT USER
# ============================================================

def current_user():

    try:

        response = (
            get_supabase()
            .auth
            .get_user()
        )

        if response and response.user:

            return response.user

    except Exception:

        pass

    return None


# ============================================================
# NEW PATIENT ID
# ============================================================

def new_patient_id():

    return (
        "MED-P-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:8].upper()
    )


# ============================================================
# RESET EXAMINATION
# ============================================================

def reset_scan():

    keys_to_reset = [
        "scan_id",
        "scan_result",
        "scan_image_bytes",
        "scan_filename",
        "review_id",
    ]

    for key in keys_to_reset:

        st.session_state[key] = None

    st.session_state.review_status = (
        "NOT_REQUESTED"
    )


# ============================================================
# RADIOLOGIST REVIEW STATUS
# ============================================================

def get_review_status(scan_id):

    if not scan_id:

        return (
            "NOT_REQUESTED",
            None,
        )

    try:

        sb = get_supabase()

        requests = (
            sb
            .table("radiologist_requests")
            .select("id,status")
            .eq(
                "scan_id",
                scan_id,
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

        reviews = (
            sb
            .table("radiologist_reviews")
            .select(
                "id,status,approved"
            )
            .eq(
                "scan_id",
                scan_id,
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

        # ----------------------------------------------------
        # APPROVED REVIEW
        # ----------------------------------------------------

        if reviews:

            review = reviews[0]

            review_status = str(
                review.get(
                    "status",
                    "",
                )
            ).upper()

            approved = (
                review.get(
                    "approved"
                )
                is True
            )

            if (
                review_status == "APPROVED"
                or approved
            ):

                return (
                    "APPROVED",
                    review.get("id"),
                )

        # ----------------------------------------------------
        # REQUEST STATUS
        # ----------------------------------------------------

        if requests:

            request_status = str(
                requests[0].get(
                    "status",
                    "",
                )
            ).upper()

            if request_status in (
                "PENDING",
                "REQUESTED",
            ):

                return (
                    "PENDING",
                    None,
                )

            if request_status in (
                "APPROVED",
                "COMPLETED",
                "REVIEWED",
            ):

                return (
                    "APPROVED",
                    None,
                )

        return (
            "NOT_REQUESTED",
            None,
        )

    except Exception:

        return (
            st.session_state.get(
                "review_status",
                "NOT_REQUESTED",
            ),
            st.session_state.get(
                "review_id"
            ),
        )


# ============================================================
# MAIN DETECTION PAGE
# ============================================================

def show_detection():

    init_state()

    set_background(
        "detection.jpg"
    )

    st.title("Medusa AI")

    st.caption(
        "AI-assisted medical imaging and "
        "radiologist review"
    )

    # ========================================================
    # PATIENT REGISTRATION
    # ========================================================

    st.subheader(
        "Patient Registration"
    )

    c1, c2 = st.columns(2)

    with c1:

        patient_name = st.text_input(
            "Patient full name",
            placeholder=(
                "Enter patient's full name"
            ),
            key="patient_name_input",
        ).strip()

    with c2:

        default_state_index = 0

        if "Delta" in STATES:

            default_state_index = (
                STATES.index("Delta")
            )

        patient_state = st.selectbox(
            "State",
            STATES,
            index=default_state_index,
            key="patient_state_input",
        )

    # --------------------------------------------------------
    # CREATE PATIENT ID
    # --------------------------------------------------------

    if (
        patient_name
        and not st.session_state.patient_id
    ):

        st.session_state.patient_id = (
            new_patient_id()
        )

    if st.session_state.patient_id:

        st.info(
            "Patient ID: "
            f"{st.session_state.patient_id}"
        )

    # ========================================================
    # NEW EXAMINATION
    # ========================================================

    if (
        st.session_state.scan_result
        is not None
    ):

        if st.button(
            "Start New Examination",
            use_container_width=True,
            key="start_new_examination",
        ):

            reset_scan()

            st.rerun()

    # ========================================================
    # EXAMINATION
    # ========================================================

    st.subheader(
        "Examination"
    )

    model_options = [
        "MammoSense — Breast Ultrasound",
        "MammoSense Pneumonia — Chest X-ray",
        "MammoSense Tuberculosis — Chest X-ray",
        "MammoSense Brain — MRI",
    ]

    model_choice = st.selectbox(
        "AI model",
        model_options,
        key="medical_model_input",
    )

    # ========================================================
    # MODEL FLAGS
    # ========================================================

    pneumonia = (
        model_choice
        == "MammoSense Pneumonia — Chest X-ray"
    )

    tuberculosis = (
        model_choice
        == "MammoSense Tuberculosis — Chest X-ray"
    )

    mammosense = (
        model_choice
        == "MammoSense — Breast Ultrasound"
    )

    brain_tumor = (
        model_choice
        == "MammoSense Brain — MRI"
    )

    # ========================================================
    # EXAMINATION NAME
    # ========================================================

    if pneumonia or tuberculosis:

        examination = "Chest X-ray"

    elif brain_tumor:

        examination = "Brain MRI"

    else:

        examination = "Breast Ultrasound"

    # ========================================================
    # DATABASE MODEL NAME
    # ========================================================

    if pneumonia:

        model_name = (
            "MammoSense Pneumonia V2"
        )

    elif tuberculosis:

        model_name = (
            "MammoSense TB V13"
        )

    elif brain_tumor:

        model_name = (
            "MammoSense Brain MRI"
        )

    else:

        model_name = (
            "MammoSense V2"
        )

    st.info(
        f"Upload a {examination.lower()} "
        "for AI-assisted screening."
    )

    # ========================================================
    # STANDARD MEDICAL IMAGE UPLOAD
    # ========================================================

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

    if uploaded:

        image_bytes = uploaded.getvalue()

        st.session_state.scan_image_bytes = (
            image_bytes
        )

        st.session_state.scan_filename = (
            uploaded.name
        )

    else:

        image_bytes = (
            st.session_state.scan_image_bytes
        )

    if not image_bytes:

        st.warning(
            "Upload an image to continue."
        )

        return

    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

        st.image(
            image,
            caption=examination,
            use_container_width=True,
        )

    except Exception as error:

        st.error(
            "The uploaded image could "
            "not be opened."
        )

        with st.expander(
            "Technical details"
        ):

            st.exception(error)

        return

    # ====================================================
    # AI ANALYSIS
    # ====================================================

    if st.button(
        "Analyze Examination",
        type="primary",
        use_container_width=True,
        key="analyze_medical_scan",
    ):

        # ------------------------------------------------
        # PATIENT VALIDATION
        # ------------------------------------------------

        if not patient_name:

            st.error(
                "Enter the patient's full name first."
            )

            return

        # ------------------------------------------------
        # USER VALIDATION
        # ------------------------------------------------

        user = current_user()

        if not user:

            st.error(
                "Your login session has expired."
            )

            return

        try:

            with st.spinner(
                "Medusa AI is analyzing..."
            ):

                # ========================================
                # PNEUMONIA
                # ========================================

                if pneumonia:

                    from ai.pneumonia import (
                        load_model as load_pneumonia_model,
                        predict as predict_pneumonia,
                    )

                    load_pneumonia_model()

                    result = (
                        predict_pneumonia(
                            image
                        )
                    )

                # ========================================
                # TUBERCULOSIS
                # ========================================

                elif tuberculosis:

                    from ai.tuberculosis import (
                        load_model as load_tb_model,
                        predict as predict_tb,
                    )

                    load_tb_model()

                    result = (
                        predict_tb(
                            image
                        )
                    )

                # ========================================
                # BRAIN MRI
                # ========================================

                elif brain_tumor:

                    from ai.braintumor import (
                        load_model as load_brain_model,
                        predict as predict_brain,
                    )

                    load_brain_model()

                    result = (
                        predict_brain(
                            image
                        )
                    )

                # ========================================
                # BREAST ULTRASOUND
                # ========================================

                else:

                    from ai.mammosense import (
                        load_model as load_mammo_model,
                        predict as predict_mammo,
                    )

                    load_mammo_model()

                    result = (
                        predict_mammo(
                            image
                        )
                    )

            # =================================================
            # VALIDATE MODEL RESULT
            # =================================================

            if not isinstance(
                result,
                dict,
            ):

                st.error(
                    "The AI model returned "
                    "an invalid result."
                )

                return

            if "prediction" not in result:

                st.error(
                    "The AI model did not "
                    "return a prediction."
                )

                return

            if "confidence" not in result:

                st.error(
                    "The AI model did not "
                    "return a confidence score."
                )

                return

            # =================================================
            # SUPABASE
            # =================================================

            sb = get_supabase()

            # =================================================
            # IMAGE EXTENSION
            # =================================================

            if uploaded:

                if "." in uploaded.name:

                    extension = (
                        uploaded.name
                        .rsplit(
                            ".",
                            1,
                        )[-1]
                        .lower()
                    )

                else:

                    extension = "png"

                content_type = (
                    uploaded.type
                    or "image/png"
                )

            else:

                extension = "png"

                content_type = (
                    "image/png"
                )

            # =================================================
            # STORAGE PATH
            # =================================================

            image_path = (
                f"{user.id}/"
                f"{st.session_state.patient_id}/"
                f"{uuid.uuid4().hex}."
                f"{extension}"
            )

            # =================================================
            # UPLOAD IMAGE
            # =================================================

            (
                sb
                .storage
                .from_(
                    "mammosense-scans"
                )
                .upload(
                    image_path,
                    image_bytes,
                    {
                        "content-type":
                            content_type,
                        "upsert":
                            "false",
                    },
                )
            )

            # =================================================
            # PROBABILITIES
            # =================================================

            probabilities = result.get(
                "probabilities",
                {},
            )

            if not isinstance(
                probabilities,
                dict,
            ):

                probabilities = {}

            # =================================================
            # SAVE AI SCAN
            # =================================================

            response = (
                sb
                .table("ai_scans")
                .insert(
                    {
                        "user_id":
                            user.id,

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
                            result.get(
                                "prediction"
                            ),

                        "confidence":
                            float(
                                result.get(
                                    "confidence",
                                    0,
                                )
                            ),

                        "probabilities":
                            probabilities,

                        "image_path":
                            image_path,

                        "status":
                            "AI_COMPLETED",
                    }
                )
                .execute()
            )

            if not response.data:

                st.error(
                    "The scan could not be saved."
                )

                return

            # =================================================
            # SESSION STATE
            # =================================================

            st.session_state.scan_id = (
                response.data[0]["id"]
            )

            st.session_state.scan_result = (
                result
            )

            st.session_state.review_status = (
                "NOT_REQUESTED"
            )

            st.session_state.review_id = None

            st.success(
                "AI analysis completed."
            )

            st.rerun()

        except Exception as error:

            st.error(
                "The examination could "
                "not be analyzed."
            )

            with st.expander(
                "Technical details"
            ):

                st.exception(error)

            return

    # ========================================================
    # RESULT
    # ========================================================

    result = (
        st.session_state.scan_result
    )

    if not result:

        return

    # ========================================================
    # REVIEW STATUS
    # ========================================================

    status, review_id = (
        get_review_status(
            st.session_state.scan_id
        )
    )

    st.session_state.review_status = (
        status
    )

    st.session_state.review_id = (
        review_id
    )

    # ========================================================
    # AI SCREENING RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "AI Screening Result"
    )

    prediction = str(
        result.get(
            "prediction",
            "Unknown",
        )
    )

    try:

        confidence = float(
            result.get(
                "confidence",
                0,
            )
        )

    except Exception:

        confidence = 0.0

    c1, c2 = st.columns(2)

    # ========================================================
    # FINDING
    # ========================================================

    with c1:

        prediction_upper = (
            prediction.upper()
        )

        # ----------------------------------------------------
        # BRAIN MRI
        # ----------------------------------------------------

        if brain_tumor:

            brain_positive_classes = (
                "GLIOMA",
                "MENINGIOMA",
                "PITUITARY",
            )

            brain_positive = (
                prediction_upper
                in brain_positive_classes
            )

            if brain_positive:

                st.error(
                    f"Finding: {prediction}"
                )

            else:

                st.success(
                    f"Finding: {prediction}"
                )

        else:

            positive_findings = (
                "MALIGNANT",
                "PNEUMONIA",
                "TUBERCULOSIS",
                "TB",
                "POSITIVE",
            )

            is_positive = any(
                label in prediction_upper
                for label in positive_findings
            )

            if is_positive:

                st.error(
                    f"Finding: {prediction}"
                )

            else:

                st.success(
                    f"Finding: {prediction}"
                )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    with c2:

        st.metric(
            "AI Confidence",
            f"{confidence:.1%}",
        )

    # ========================================================
    # TB INFORMATION
    # ========================================================

    if tuberculosis:

        st.caption(
            "MammoSense TB V13 • "
            "binary chest X-ray classifier"
        )

        if prediction_upper == "TB":

            st.warning(
                "TB-positive AI screening finding. "
                "Radiologist review is required."
            )

        elif prediction_upper == "NON_TB":

            st.success(
                "AI classified this examination "
                "as NON_TB."
            )

    # ========================================================
    # BRAIN MRI INFORMATION
    # ========================================================

    if brain_tumor:

        st.caption(
            "MammoSense Brain MRI • "
            "2D ResNet-50 • 4-class classifier"
        )

        brain_positive_classes = (
            "GLIOMA",
            "MENINGIOMA",
            "PITUITARY",
        )

        if prediction_upper in brain_positive_classes:

            st.warning(
                f"AI classified the Brain MRI as "
                f"{prediction}. "
                "Radiologist review is required."
            )

        elif prediction_upper == "NOTUMOR":

            st.success(
                "AI classified this Brain MRI "
                "as no tumor."
            )

        else:

            st.info(
                "Brain MRI classification completed. "
                "Radiologist review is required."
            )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    probabilities = result.get(
        "probabilities",
        {},
    )

    if (
        isinstance(
            probabilities,
            dict,
        )
        and probabilities
    ):

        st.subheader(
            "Probability Breakdown"
        )

        for label, value in (
            probabilities.items()
        ):

            try:

                value = float(
                    value
                )

            except Exception:

                continue

            st.write(
                f"{label}: "
                f"{value:.2%}"
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
    # RADIOLOGIST REVIEW
    # ========================================================

    st.divider()

    st.subheader(
        "Radiologist Review"
    )

    st.info(
        "Every examination must be reviewed "
        "and approved by a qualified radiologist "
        "before the final medical report can "
        "be downloaded."
    )

    # ========================================================
    # APPROVED
    # ========================================================

    if status == "APPROVED":

        st.success(
            "Radiologist review completed "
            "and approved."
        )

    # ========================================================
    # PENDING
    # ========================================================

    elif status == "PENDING":

        st.warning(
            "Examination is waiting for "
            "radiologist review."
        )

    # ========================================================
    # NOT REQUESTED
    # ========================================================

    else:

        st.warning(
            "Radiologist review has "
            "not been requested."
        )

        if st.button(
            "Submit for Radiologist Review",
            type="primary",
            use_container_width=True,
            key="submit_radiologist_request",
        ):

            user = current_user()

            if not user:

                st.error(
                    "Please log in again."
                )

                return

            try:

                sb = get_supabase()

                scan_id = (
                    st.session_state.scan_id
                )

                if not scan_id:

                    st.error(
                        "No completed scan was found."
                    )

                    return

                # ==========================================
                # CHECK EXISTING REQUEST
                # ==========================================

                existing = (
                    sb
                    .table(
                        "radiologist_requests"
                    )
                    .select(
                        "id,status"
                    )
                    .eq(
                        "scan_id",
                        scan_id,
                    )
                    .limit(1)
                    .execute()
                    .data
                    or []
                )

                if existing:

                    existing_status = str(
                        existing[0].get(
                            "status",
                            "",
                        )
                    ).upper()

                    if existing_status in (
                        "PENDING",
                        "REQUESTED",
                    ):

                        st.session_state.review_status = (
                            "PENDING"
                        )

                        st.info(
                            "This examination is already "
                            "waiting for review."
                        )

                        st.rerun()

                    if existing_status in (
                        "APPROVED",
                        "COMPLETED",
                        "REVIEWED",
                    ):

                        st.session_state.review_status = (
                            "APPROVED"
                        )

                        st.rerun()

                # ==========================================
                # CREATE REQUEST
                # ==========================================

                request = (
                    sb
                    .table(
                        "radiologist_requests"
                    )
                    .insert(
                        {
                            "user_id":
                                user.id,

                            "scan_id":
                                scan_id,

                            "status":
                                "PENDING",
                        }
                    )
                    .execute()
                )

                if not request.data:

                    st.error(
                        "The review request "
                        "could not be created."
                    )

                    return

                # ==========================================
                # UPDATE SCAN STATUS
                # ==========================================

                (
                    sb
                    .table("ai_scans")
                    .update(
                        {
                            "status":
                                "AWAITING_RADIOLOGIST"
                        }
                    )
                    .eq(
                        "id",
                        scan_id,
                    )
                    .execute()
                )

                st.session_state.review_status = (
                    "PENDING"
                )

                st.success(
                    "Examination successfully "
                    "submitted for radiologist review."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "Could not submit the "
                    "examination for review."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.exception(error)

    # ========================================================
    # FINAL MEDICAL REPORT
    # ========================================================

    st.divider()

    st.subheader(
        "Final Medical Report"
    )

    if status != "APPROVED":

        st.error(
            "DOWNLOAD LOCKED"
        )

        st.caption(
            "The final medical report becomes "
            "available only after radiologist "
            "review and approval."
        )

        return

    st.success(
        "Radiologist-approved report available."
    )

    # ========================================================
    # LOAD APPROVED REPORT
    # ========================================================

    try:

        sb = get_supabase()

        reports = (
            sb
            .table("medical_reports")
            .select(
                "report_id,pdf_path,status"
            )
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

        if not reports:

            st.info(
                "The examination has been approved. "
                "The final report is being prepared."
            )

            return

        report = reports[0]

        pdf_path = report.get(
            "pdf_path"
        )

        report_id = (
            report.get(
                "report_id"
            )
            or "MEDUSA_REPORT"
        )

        if not pdf_path:

            st.warning(
                "The report exists but its "
                "PDF file is not available yet."
            )

            return

        # ====================================================
        # DOWNLOAD PDF
        # ====================================================

        pdf = (
            sb
            .storage
            .from_(
                "medical-reports"
            )
            .download(
                pdf_path
            )
        )

        st.download_button(
            "Download Final Medical Report",
            data=pdf,
            file_name=f"{report_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            key=(
                f"detection_download_report_"
                f"{report_id}"
            ),
        )

    except Exception as error:

        st.error(
            "The approved report could "
            "not be loaded."
        )

        with st.expander(
            "Technical details"
        ):

            st.exception(error)

    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "Medusa AI provides AI-assisted "
        "screening and does not replace "
        "professional medical diagnosis "
        "or treatment."
    )
