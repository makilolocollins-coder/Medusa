# ================================================================
# MEDUSA AI
# DETECTION + PATIENT REGISTRATION + RADIOLOGIST REVIEW
# + APPROVED MEDICAL REPORT
#
# IMPORTANT:
# PDF DOWNLOAD IS LOCKED UNTIL RADIOLOGIST APPROVAL.
# ================================================================

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
        "review_status": "Not requested",
        "review_id": None,
        "report_pdf": None,
        "report_id": None,
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
# GET CURRENT USER
# ================================================================

def get_current_user():

    supabase = get_supabase()

    response = supabase.auth.get_user()

    if not response.user:
        return None

    return response.user


# ================================================================
# RESET CURRENT EXAMINATION
# ================================================================

def reset_examination():

    st.session_state.scan_id = None
    st.session_state.scan_model = None
    st.session_state.scan_type = None
    st.session_state.scan_result = None
    st.session_state.scan_image_bytes = None
    st.session_state.scan_filename = None
    st.session_state.review_status = "Not requested"
    st.session_state.review_id = None
    st.session_state.report_pdf = None
    st.session_state.report_id = None
    st.session_state.report_downloadable = False


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
    # GENERATE PATIENT ID
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

        uploader_label = (
            "Upload chest X-ray"
        )

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
    # NEW EXAMINATION BUTTON
    # ============================================================

    if st.session_state.scan_result is not None:

        if st.button(
            "＋ Start New Examination",
            use_container_width=True,
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

        # Persist image across Streamlit reruns
        st.session_state.scan_image_bytes = (
            image_bytes
        )

        st.session_state.scan_filename = (
            uploaded.name
        )

        try:

            image = Image.open(
                uploaded
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

        image = Image.open(
            __import__("io").BytesIO(
                image_bytes
            )
        ).convert("RGB")

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
            # SAVE IMAGE
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
            # ----------------------------------------------------

            scan_response = (
                supabase
                .table("ai_scans")
                .insert({
                    "user_id": user_id,
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

            if not scan_response.data:

                st.error(
                    "AI analysis completed, but "
                    "the scan could not be saved."
                )

                return

            scan_id = scan_response.data[0]["id"]

            # ----------------------------------------------------
            # SESSION STATE
            # ----------------------------------------------------

            st.session_state.scan_result = result

            st.session_state.scan_id = scan_id

            st.session_state.scan_model = model_name

            st.session_state.scan_type = (
                "pneumonia"
                if is_pneumonia
                else "mammosense"
            )

            st.session_state.review_status = (
                "Not requested"
            )

            st.session_state.review_id = None

            st.session_state.report_pdf = None

            st.session_state.report_id = None

            st.session_state.report_downloadable = (
                False
            )

            st.success(
                "✅ AI analysis completed."
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

    st.subheader("🤖 AI Screening Result")

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
    # RADIOLOGIST REVIEW
    # ============================================================

    st.divider()

    st.subheader(
        "👨‍⚕️ Radiologist Review"
    )

    st.warning(
        "AI output is preliminary and is not "
        "a final medical diagnosis. A qualified "
        "radiologist must review the examination "
        "before a final report can be generated."
    )

    # ------------------------------------------------------------
    # CHECK CURRENT REVIEW
    # ------------------------------------------------------------

    try:

        supabase = get_supabase()

        existing_reviews = (
            supabase
            .table("radiologist_reviews")
            .select("*")
            .eq(
                "scan_id",
                st.session_state.scan_id,
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

    except Exception:

        existing_reviews = []

    if existing_reviews:

        review = existing_reviews[0]

        st.session_state.review_id = review.get(
            "id"
        )

        st.session_state.review_status = (
            review.get(
                "status",
                "Pending",
            )
        )

    status = st.session_state.review_status

    if status == "APPROVED":

        st.success(
            "✅ Radiologist review approved."
        )

    elif status == "PENDING":

        st.info(
            "⏳ Radiologist review is pending."
        )

    else:

        st.info(
            "No radiologist review has been "
            "submitted for this examination."
        )

    # ============================================================
    # REVIEW FORM
    # ============================================================

    if status != "APPROVED":

        radiologist_name = st.text_input(
            "Radiologist name",
            key="radiologist_name",
        )

        registration_number = st.text_input(
            "Medical registration number",
            key="registration_number",
        )

        findings = st.text_area(
            "Findings",
            placeholder=(
                "Describe the radiographic findings..."
            ),
            height=160,
            key="radiologist_findings",
        )

        impression = st.text_area(
            "Impression",
            placeholder=(
                "Enter the final radiologist impression..."
            ),
            height=120,
            key="radiologist_impression",
        )

        recommendations = st.text_area(
            "Recommendations",
            placeholder=(
                "Enter recommendations if applicable..."
            ),
            height=100,
            key="radiologist_recommendations",
        )

        remarks = st.text_area(
            "Radiologist remarks",
            placeholder=(
                "Additional remarks..."
            ),
            height=100,
            key="radiologist_remarks",
        )

        approval = st.checkbox(
            "I have personally reviewed the examination "
            "and approve the clinical interpretation "
            "entered above.",
            key="radiologist_approval",
        )

        if st.button(
            "✓ Approve Review & Generate Final Report",
            type="primary",
            use_container_width=True,
            key="approve_generate_report",
        ):

            # ----------------------------------------------------
            # VALIDATION
            # ----------------------------------------------------

            if not radiologist_name.strip():

                st.error(
                    "Radiologist name is required."
                )

                return

            if not registration_number.strip():

                st.error(
                    "Medical registration number "
                    "is required."
                )

                return

            if not findings.strip():

                st.error(
                    "Radiologist findings are required."
                )

                return

            if not impression.strip():

                st.error(
                    "Radiologist impression is required."
                )

                return

            if not approval:

                st.error(
                    "The radiologist must explicitly "
                    "approve the examination."
                )

                return

            try:

                user = get_current_user()

                if user is None:

                    st.error(
                        "Please log in again."
                    )

                    return

                supabase = get_supabase()

                reviewed_at = datetime.now()

                # ------------------------------------------------
                # SAVE RADIOLOGIST REVIEW
                # ------------------------------------------------

                review_response = (
                    supabase
                    .table("radiologist_reviews")
                    .insert({
                        "scan_id": (
                            st.session_state.scan_id
                        ),
                        "user_id": user.id,
                        "radiologist_name":
                            radiologist_name.strip(),
                        "registration_number":
                            registration_number.strip(),
                        "findings":
                            findings.strip(),
                        "impression":
                            impression.strip(),
                        "recommendations":
                            recommendations.strip(),
                        "remarks":
                            remarks.strip(),
                        "status":
                            "APPROVED",
                        "approved":
                            True,
                        "reviewed_at":
                            reviewed_at.isoformat(),
                    })
                    .execute()
                )

                if not review_response.data:

                    st.error(
                        "The radiologist review "
                        "could not be saved."
                    )

                    return

                review_id = (
                    review_response.data[0]["id"]
                )

                # ------------------------------------------------
                # UPDATE SCAN
                # ------------------------------------------------

                (
                    supabase
                    .table("ai_scans")
                    .update({
                        "status":
                            "RADIOLOGIST_APPROVED",
                    })
                    .eq(
                        "id",
                        st.session_state.scan_id,
                    )
                    .execute()
                )

                # ------------------------------------------------
                # GENERATE FINAL PDF
                # ------------------------------------------------

                report_buffer, report_id = (
                    generate_pdf_report(
                        patient_name=patient_name,
                        patient_id=(
                            st.session_state.patient_id
                        ),
                        state=patient_state,
                        examination=examination,
                        ai_prediction=prediction,
                        ai_confidence=confidence,
                        probabilities=probabilities,
                        radiologist_name=(
                            radiologist_name.strip()
                        ),
                        registration_number=(
                            registration_number.strip()
                        ),
                        findings=findings.strip(),
                        impression=impression.strip(),
                        recommendations=(
                            recommendations.strip()
                        ),
                        remarks=remarks.strip(),
                        reviewed_at=(
                            reviewed_at.strftime(
                                "%d %B %Y, %H:%M"
                            )
                        ),
                        xray_image=image_bytes
                        if is_pneumonia
                        else None,
                        ultrasound_image=image_bytes
                        if not is_pneumonia
                        else None,
                    )
                )

                pdf_bytes = (
                    report_buffer.getvalue()
                )

                # ------------------------------------------------
                # SAVE PDF
                # ------------------------------------------------

                pdf_path = (
                    f"{user.id}/"
                    f"{st.session_state.patient_id}/"
                    f"{report_id}.pdf"
                )

                supabase.storage.from_(
                    "medical-reports"
                ).upload(
                    pdf_path,
                    pdf_bytes,
                    {
                        "content-type":
                            "application/pdf",
                        "upsert": "false",
                    },
                )

                # ------------------------------------------------
                # SAVE REPORT RECORD
                # ------------------------------------------------

                report_response = (
                    supabase
                    .table("medical_reports")
                    .insert({
                        "report_id":
                            report_id,
                        "scan_id":
                            st.session_state.scan_id,
                        "review_id":
                            review_id,
                        "user_id":
                            user.id,
                        "patient_id":
                            st.session_state.patient_id,
                        "patient_name":
                            patient_name,
                        "patient_state":
                            patient_state,
                        "status":
                            "APPROVED",
                        "pdf_path":
                            pdf_path,
                        "approved_at":
                            reviewed_at.isoformat(),
                    })
                    .execute()
                )

                if not report_response.data:

                    st.error(
                        "The PDF was generated, "
                        "but the report record could "
                        "not be saved."
                    )

                    return

                # ------------------------------------------------
                # UNLOCK DOWNLOAD
                # ------------------------------------------------

                st.session_state.review_id = review_id

                st.session_state.review_status = (
                    "APPROVED"
                )

                st.session_state.report_pdf = (
                    pdf_bytes
                )

                st.session_state.report_id = (
                    report_id
                )

                st.session_state.report_downloadable = (
                    True
                )

                st.success(
                    "✅ Radiologist review approved."
                )

                st.success(
                    "✅ Final medical report generated."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "The report could not be generated."
                )

                st.exception(error)

    # ============================================================
    # FINAL REPORT
    # ============================================================

    st.divider()

    st.subheader(
        "📄 Final Medical Report"
    )

    # ------------------------------------------------------------
    # HARD DOWNLOAD LOCK
    # ------------------------------------------------------------

    if (
        st.session_state.review_status
        == "APPROVED"
        and st.session_state.report_downloadable
        and st.session_state.report_pdf
        and st.session_state.report_id
    ):

        st.success(
            f"✓ Final report approved\n\n"
            f"Report ID: "
            f"**{st.session_state.report_id}**"
        )

        st.download_button(
            label="⬇️ Download Final Medical Report",
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

        st.error(
            "🔒 DOWNLOAD LOCKED"
        )

        st.caption(
            "The final PDF cannot be downloaded "
            "until a radiologist has reviewed and "
            "approved this examination."
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
