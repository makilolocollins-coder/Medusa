# ================================================================
# MEDUSA AI
# DETECTION + PATIENT REGISTRATION + MANDATORY RADIOLOGIST REVIEW
# + FINAL MEDICAL REPORT
#
# HARD RULE:
# Every scan must have an APPROVED radiologist review before
# the PDF can be downloaded.
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

from reports.pdf_report import generate_pdf_report
from reports.report_guard import (
    has_approved_review,
    get_approved_review,
)


# ================================================================
# CONFIG
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
        "patient_state": "Abia",

        "scan_id": None,
        "scan_model": None,
        "scan_type": None,
        "scan_result": None,

        "scan_image_bytes": None,
        "scan_filename": None,

        "review_status": "NOT_REVIEWED",
        "review_id": None,
        "review_data": None,

        "report_id": None,
        "report_pdf": None,

        "new_scan_nonce": 0,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ================================================================
# PATIENT ID
# ================================================================

def generate_patient_id():

    return (
        "MED-P-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:8].upper()
    )


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
# RESET SCAN ONLY
#
# Patient identity remains.
# A new scan gets a completely new scan_id.
# ================================================================

def reset_scan():

    st.session_state.scan_id = None
    st.session_state.scan_model = None
    st.session_state.scan_type = None
    st.session_state.scan_result = None

    st.session_state.scan_image_bytes = None
    st.session_state.scan_filename = None

    st.session_state.review_status = "NOT_REVIEWED"
    st.session_state.review_id = None
    st.session_state.review_data = None

    st.session_state.report_id = None
    st.session_state.report_pdf = None

    st.session_state.new_scan_nonce += 1


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
    # PATIENT REGISTRATION
    # ============================================================

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        patient_name = st.text_input(
            "Patient full name",
            value=st.session_state.patient_name,
            placeholder="Enter patient's full name",
            key="patient_name_input",
        ).strip()

    with col2:

        patient_state = st.selectbox(
            "State",
            NIGERIAN_STATES,
            index=NIGERIAN_STATES.index(
                st.session_state.patient_state
            )
            if st.session_state.patient_state
            in NIGERIAN_STATES
            else 0,
            key="patient_state_input",
        )

    st.session_state.patient_name = patient_name
    st.session_state.patient_state = patient_state

    if patient_name and not st.session_state.patient_id:

        st.session_state.patient_id = (
            generate_patient_id()
        )

    if st.session_state.patient_id:

        st.success(
            "Patient ID: "
            + st.session_state.patient_id
        )

    # ============================================================
    # NEW EXAMINATION
    # ============================================================

    if st.session_state.scan_id:

        if st.button(
            "＋ Start New Scan",
            use_container_width=True,
        ):

            reset_scan()

            st.rerun()

    # ============================================================
    # MODEL
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
        button_label = "🔬 Analyze Chest X-ray"

        st.info(
            "Upload a chest X-ray for "
            "AI-assisted pneumonia screening."
        )

    else:

        examination = "Breast Ultrasound"
        model_name = "MammoSense V2"
        uploader_label = "Upload breast ultrasound"
        button_label = "🔬 Analyze Breast Ultrasound"

        st.info(
            "Upload a breast ultrasound for "
            "AI-assisted screening."
        )

    # ============================================================
    # UPLOAD
    # ============================================================

    uploaded = st.file_uploader(
        uploader_label,
        type=["jpg", "jpeg", "png", "webp"],
        key=f"medical_image_{st.session_state.new_scan_nonce}",
    )

    image = None
    image_bytes = None

    if uploaded is not None:

        image_bytes = uploaded.getvalue()

        st.session_state.scan_image_bytes = image_bytes
        st.session_state.scan_filename = uploaded.name

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
                "The saved scan image could not be opened."
            )

            return

        st.image(
            image,
            caption=examination,
            use_container_width=True,
        )

    else:

        st.warning(
            "Upload an examination image to continue."
        )

        return

    # ============================================================
    # ANALYZE
    # ============================================================

    if st.button(
        button_label,
        type="primary",
        use_container_width=True,
        key="analyze_scan",
    ):

        if not patient_name:

            st.error(
                "Patient full name is required."
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

            # ----------------------------------------------------
            # IMAGE STORAGE
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
            # SAVE SCAN
            # ----------------------------------------------------

            scan_data = {
                "user_id": user.id,
                "patient_id": st.session_state.patient_id,
                "patient_name": patient_name,
                "patient_state": patient_state,
                "examination": examination,
                "model": model_name,
                "prediction": result.get(
                    "prediction",
                    "Unknown",
                ),
                "confidence": float(
                    result.get("confidence", 0)
                ),
                "probabilities": result.get(
                    "probabilities",
                    {},
                ),
                "image_path": image_path,
                "status": "AI_COMPLETED",
            }

            response = (
                supabase
                .table("ai_scans")
                .insert(scan_data)
                .execute()
            )

            if not response.data:

                st.error(
                    "The AI result could not be saved."
                )

                return

            scan_id = response.data[0]["id"]

            # ----------------------------------------------------
            # RESET REVIEW STATE FOR THIS NEW SCAN
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
                "NOT_REVIEWED"
            )
            st.session_state.review_id = None
            st.session_state.review_data = None

            st.session_state.report_id = None
            st.session_state.report_pdf = None

            st.success(
                "✅ AI analysis completed."
            )

            st.info(
                "🔒 Final report download is locked "
                "until this specific scan is reviewed "
                "and approved by a radiologist."
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

    st.divider()

    st.subheader("🤖 AI Screening Result")

    st.caption(
        f"Scan ID: {st.session_state.scan_id}"
    )

    st.caption(
        f"Model: {st.session_state.scan_model}"
    )

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
                min(max(value, 0.0), 1.0)
            )

    # ============================================================
    # VERIFY REVIEW FROM DATABASE
    #
    # DO NOT TRUST SESSION STATE.
    # ============================================================

    scan_id = st.session_state.scan_id

    try:

        approved_review = get_approved_review(
            scan_id
        )

    except Exception:

        approved_review = None

    if approved_review:

        st.session_state.review_status = "APPROVED"
        st.session_state.review_id = (
            approved_review.get("id")
        )
        st.session_state.review_data = (
            approved_review
        )

    else:

        st.session_state.review_status = (
            "NOT_REVIEWED"
        )

    # ============================================================
    # RADIOLOGIST REVIEW
    # ============================================================

    st.divider()

    st.subheader(
        "👨‍⚕️ Mandatory Radiologist Review"
    )

    st.warning(
        "This examination cannot receive a "
        "downloadable final report until this "
        "specific scan has been reviewed and "
        "approved by a radiologist."
    )

    # ============================================================
    # APPROVED
    # ============================================================

    if st.session_state.review_status == "APPROVED":

        review = st.session_state.review_data

        st.success(
            "✅ THIS SCAN HAS BEEN REVIEWED AND APPROVED."
        )

        st.write(
            f"**Radiologist:** "
            f"{review.get('radiologist_name', 'N/A')}"
        )

        st.write(
            f"**Registration:** "
            f"{review.get('registration_number', 'N/A')}"
        )

        st.write(
            f"**Reviewed:** "
            f"{review.get('reviewed_at', 'N/A')}"
        )

        with st.expander(
            "View radiologist review"
        ):

            st.write(
                "**Findings**"
            )
            st.write(
                review.get("findings", "")
            )

            st.write(
                "**Impression**"
            )
            st.write(
                review.get("impression", "")
            )

            st.write(
                "**Recommendations**"
            )
            st.write(
                review.get("recommendations", "")
            )

            st.write(
                "**Remarks**"
            )
            st.write(
                review.get("remarks", "")
            )

    # ============================================================
    # REVIEW FORM
    # ============================================================

    else:

        radiologist_name = st.text_input(
            "Radiologist name",
            key="radiologist_name",
        )

        registration_number = st.text_input(
            "Medical registration number",
            key="registration_number",
        )

        findings = st.text_area(
            "Findings *",
            height=160,
            placeholder=(
                "Describe the radiological findings..."
            ),
            key="radiologist_findings",
        )

        impression = st.text_area(
            "Impression *",
            height=120,
            placeholder=(
                "Enter the radiologist's final impression..."
            ),
            key="radiologist_impression",
        )

        recommendations = st.text_area(
            "Recommendations",
            height=100,
            placeholder=(
                "Enter recommendations if applicable..."
            ),
            key="radiologist_recommendations",
        )

        remarks = st.text_area(
            "Radiologist remarks",
            height=100,
            placeholder=(
                "Additional professional remarks..."
            ),
            key="radiologist_remarks",
        )

        approval = st.checkbox(
            "I confirm that I have personally reviewed "
            "THIS examination and approve the clinical "
            "interpretation entered above.",
            key="radiologist_approval",
        )

        if st.button(
            "✓ Complete Radiologist Review",
            type="primary",
            use_container_width=True,
            key="complete_radiologist_review",
        ):

            if not radiologist_name.strip():

                st.error(
                    "Radiologist name is required."
                )

                return

            if not registration_number.strip():

                st.error(
                    "Medical registration number is required."
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
                    "approve this examination."
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
                # SAVE REVIEW FOR THIS EXACT SCAN
                # ------------------------------------------------

                review_response = (
                    supabase
                    .table("radiologist_reviews")
                    .insert({
                        "scan_id": scan_id,
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
                # UPDATE EXACT SCAN
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
                        scan_id,
                    )
                    .execute()
                )

                # ------------------------------------------------
                # GENERATE REPORT ONLY AFTER APPROVAL
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
                        xray_image=(
                            image_bytes
                            if is_pneumonia
                            else None
                        ),
                        ultrasound_image=(
                            image_bytes
                            if not is_pneumonia
                            else None
                        ),
                    )
                )

                pdf_bytes = (
                    report_buffer.getvalue()
                )

                # ------------------------------------------------
                # STORE REPORT
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
                # REPORT RECORD
                # ------------------------------------------------

                report_response = (
                    supabase
                    .table("medical_reports")
                    .insert({
                        "report_id":
                            report_id,
                        "scan_id":
                            scan_id,
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
                        "The report could not be registered."
                    )

                    return

                # ------------------------------------------------
                # SAVE TO SESSION ONLY AFTER DATABASE SUCCESS
                # ------------------------------------------------

                st.session_state.review_status = (
                    "APPROVED"
                )

                st.session_state.review_id = (
                    review_id
                )

                st.session_state.report_id = (
                    report_id
                )

                st.session_state.report_pdf = (
                    pdf_bytes
                )

                st.session_state.review_data = {
                    "id": review_id,
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
                    "reviewed_at":
                        reviewed_at.isoformat(),
                }

                st.success(
                    "✅ Radiologist review completed."
                )

                st.success(
                    "✅ Final medical report generated."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "The radiologist review/report "
                    "could not be completed."
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
    # HARD DATABASE CHECK AGAIN
    # ------------------------------------------------------------

    database_approved = False

    try:

        database_approved = has_approved_review(
            scan_id
        )

    except Exception:

        database_approved = False

    # ------------------------------------------------------------
    # DOWNLOAD
    # ------------------------------------------------------------

    if (
        database_approved
        and st.session_state.report_pdf
        and st.session_state.report_id
    ):

        st.success(
            "✓ Radiologist-approved final report"
        )

        st.caption(
            f"Report ID: "
            f"{st.session_state.report_id}"
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
            key="final_report_download",
        )

    else:

        st.error(
            "🔒 FINAL REPORT DOWNLOAD LOCKED"
        )

        st.caption(
            "A final report cannot be downloaded "
            "until this exact scan has an APPROVED "
            "radiologist review."
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
