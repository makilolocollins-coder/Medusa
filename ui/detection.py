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

from ai.tuberculosis import (
    load_model as load_tb_model,
    predict as predict_tb,
)

from ai.braintumor import (
    load_model as load_brain_model,
    predict as predict_brain,
)




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

        return response.user

    except Exception:

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

    for key in [
        "scan_id",
        "scan_result",
        "scan_image_bytes",
        "scan_filename",
        "review_id",
    ]:

        st.session_state[key] = None

    st.session_state.review_status = "NOT_REQUESTED"


# ============================================================
# RADIOLOGIST REVIEW STATUS
# ============================================================

def review_status(scan_id):

    if not scan_id:

        return "NOT_REQUESTED", None

    sb = get_supabase()

    try:

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

        # --------------------------------------------------------
        # APPROVED REVIEW
        # --------------------------------------------------------

        if reviews:

            r = reviews[0]

            if (
                str(
                    r.get(
                        "status",
                        "",
                    )
                ).upper()
                == "APPROVED"
                or r.get("approved") is True
            ):

                return (
                    "APPROVED",
                    r.get("id"),
                )

        # --------------------------------------------------------
        # REQUEST STATUS
        # --------------------------------------------------------

        if requests:

            status = str(
                requests[0].get(
                    "status",
                    "",
                )
            ).upper()

            if status in (
                "PENDING",
                "REQUESTED",
            ):

                return (
                    "PENDING",
                    None,
                )

            if status in (
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
        "AI-assisted medical imaging and radiologist review"
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

        patient_state = st.selectbox(
            "State",
            STATES,
            index=STATES.index("Delta"),
            key="patient_state_input",
        )

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
    # EXAMINATION TYPE
    # ========================================================

    st.subheader(
        "Examination"
    )

    model_choice = st.selectbox(
        "AI model",
        [
            "MammoSense — Breast Ultrasound",
            "MammoSense Pneumonia — Chest X-ray",
            "MammoSense Tuberculosis — Chest X-ray",
            "MammoSense Brain — MRI",
        ],
        key="medical_model_input",
    )

    # --------------------------------------------------------
    # MODEL FLAGS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EXAMINATION NAME
    # --------------------------------------------------------

    if pneumonia or tuberculosis:

        examination = "Chest X-ray"
        
    elif brain_tumor:

    examination = "Brain MRI"
    
    else:

        examination = "Breast Ultrasound"

    # --------------------------------------------------------
    # MODEL NAME STORED IN DATABASE
    # --------------------------------------------------------

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
           "MammoSense Brain V3.1"
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
# BRAIN MRI UPLOAD AND OTHER SCANS 
# ========================================================

if brain_tumor:

    st.info(
        "Upload the four MRI volumes required by "
        "MammoSense Brain V3.1: T1, T1CE, T2 and FLAIR."
    )

    t1_file = st.file_uploader(
        "T1 MRI",
        type=["nii", "gz"],
        key="brain_t1_upload",
    )

    t1ce_file = st.file_uploader(
        "T1CE MRI",
        type=["nii", "gz"],
        key="brain_t1ce_upload",
    )

    t2_file = st.file_uploader(
        "T2 MRI",
        type=["nii", "gz"],
        key="brain_t2_upload",
    )

    flair_file = st.file_uploader(
        "T2-FLAIR MRI",
        type=["nii", "gz"],
        key="brain_flair_upload",
    )

    if not all([
        t1_file,
        t1ce_file,
        t2_file,
        flair_file,
    ]):

        st.warning(
            "Upload all four MRI volumes "
            "before continuing."
        )

        return

    uploaded = None

    image_bytes = None

else:

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

    # ========================================================
    # OPEN IMAGE
    # ========================================================

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

        st.exception(error)

        return

    # ========================================================
    # AI ANALYSIS
    # ========================================================

    if st.button(
        "Analyze Examination",
        type="primary",
        use_container_width=True,
        key="analyze_medical_scan",
    ):

        # ----------------------------------------------------
        # PATIENT VALIDATION
        # ----------------------------------------------------

        if not patient_name:

            st.error(
                "Enter the patient's full name first."
            )

            return

        # ----------------------------------------------------
        # USER VALIDATION
        # ----------------------------------------------------

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

                # ==============================================
                # PNEUMONIA
                # ==============================================

                if pneumonia:

                    load_pneumonia_model()

                    result = (
                        predict_pneumonia(
                            image
                        )
                    )

                # ==============================================
                # TUBERCULOSIS
                # ==============================================

                elif tuberculosis:

                    load_tb_model()

                    result = (
                        predict_tb(
                            image
                        )
                    )

                # ==============================================
                # BREAST ULTRASOUND
                # ==============================================

                else:

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

            extension = (
                uploaded.name.rsplit(
                    ".",
                    1,
                )[-1].lower()
                if uploaded
                else "png"
            )

            content_type = (
                uploaded.type
                if uploaded
                else "image/png"
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
                            result.get(
                                "probabilities",
                                {},
                            ),

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

    status, review_id = review_status(
        st.session_state.scan_id
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
    # TB-SPECIFIC INFORMATION
    # ========================================================

    if tuberculosis:

        st.caption(
    "MammoSense TB V13 • "
    "binary chest X-ray classifier"
)

        if (
            prediction.upper()
            == "TB"
        ):

            st.warning(
                "TB-positive AI screening finding. "
                "Radiologist review is required."
            )

        elif (
            prediction.upper()
            == "NON_TB"
        ):

            st.success(
                "AI classified this examination "
                "as NON_TB."
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

                    st.session_state.review_status = (
                        "PENDING"
                    )

                    st.info(
                        "This examination is already "
                        "waiting for review."
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

        pdf_path = (
            report.get(
                "pdf_path"
            )
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
