# ================================================================
# MEDUSA AI
# RADIOLOGIST DASHBOARD
#
# ONLY RADIOLOGIST SIDE
#
# Radiologist:
#   - sees pending scans
#   - reviews image
#   - enters findings
#   - enters impression
#   - enters recommendations
#   - enters remarks
#   - approves report
#
# Approval generates final PDF.
# ================================================================

import io
from datetime import datetime
import uuid

import streamlit as st
from PIL import Image

from utils.supabase_client import get_supabase
from reports.pdf_report import generate_pdf_report


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
# RADIOLOGIST DASHBOARD
# ================================================================

def show_radiologist():

    st.title("👨‍⚕️ Radiologist Dashboard")

    st.caption(
        "Medusa AI — Clinical Review Centre"
    )

    user = get_current_user()

    if user is None:

        st.error(
            "Please log in to continue."
        )

        return

    supabase = get_supabase()

    # ============================================================
    # PENDING SCANS
    # ============================================================

    st.subheader(
        "Pending Examinations"
    )

    try:

        pending = (
            supabase
            .table("ai_scans")
            .select("*")
            .eq(
                "status",
                "PENDING_REVIEW",
            )
            .order(
                "created_at",
                desc=False,
            )
            .execute()
            .data
            or []
        )

    except Exception as error:

        st.error(
            "Could not load pending examinations."
        )

        st.exception(error)

        return

    if not pending:

        st.success(
            "✓ No examinations are currently "
            "waiting for review."
        )

        return

    st.info(
        f"{len(pending)} examination(s) "
        "awaiting review."
    )

    # ============================================================
    # SELECT SCAN
    # ============================================================

    scan_labels = []

    scan_map = {}

    for scan in pending:

        label = (
            f"{scan.get('patient_name', 'Unknown')} "
            f"| "
            f"{scan.get('patient_id', 'No ID')} "
            f"| "
            f"{scan.get('examination', 'Examination')}"
        )

        scan_labels.append(label)

        scan_map[label] = scan

    selected_label = st.selectbox(
        "Select examination",
        scan_labels,
        key="radiologist_scan_selector",
    )

    scan = scan_map[selected_label]

    scan_id = scan["id"]

    # ============================================================
    # PATIENT INFORMATION
    # ============================================================

    st.divider()

    st.subheader(
        "Patient Information"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Patient",
            scan.get(
                "patient_name",
                "Unknown",
            ),
        )

    with c2:

        st.metric(
            "Patient ID",
            scan.get(
                "patient_id",
                "Unknown",
            ),
        )

    with c3:

        st.metric(
            "State",
            scan.get(
                "patient_state",
                "Unknown",
            ),
        )

    st.write(
        f"**Examination:** "
        f"{scan.get('examination', 'Unknown')}"
    )

    st.write(
        f"**AI Model:** "
        f"{scan.get('model', 'Unknown')}"
    )

    # ============================================================
    # IMAGE
    # ============================================================

    st.divider()

    st.subheader(
        "Medical Image"
    )

    image_bytes = None

    image_path = scan.get(
        "image_path"
    )

    if image_path:

        try:

            image_bytes = (
                supabase
                .storage
                .from_("mammosense-scans")
                .download(image_path)
            )

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            st.image(
                image,
                caption=scan.get(
                    "examination",
                    "Medical image",
                ),
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                "Unable to load the medical image."
            )

            st.exception(error)

    # ============================================================
    # AI RESULT
    # ============================================================

    st.divider()

    st.subheader(
        "AI Screening Result"
    )

    prediction = scan.get(
        "prediction",
        "Unknown",
    )

    confidence = float(
        scan.get(
            "confidence",
            0,
        )
    )

    c1, c2 = st.columns(2)

    with c1:

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

    with c2:

        st.metric(
            "AI Confidence",
            f"{confidence:.1%}",
        )

    probabilities = scan.get(
        "probabilities",
        {},
    )

    if isinstance(probabilities, dict):

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
        "Clinical Review"
    )

    st.warning(
        "The AI result is an assistive screening "
        "output. The radiologist must independently "
        "review the medical image and provide the "
        "final clinical interpretation."
    )

    radiologist_name = st.text_input(
        "Radiologist full name",
        key=f"rad_name_{scan_id}",
    )

    registration_number = st.text_input(
        "Medical registration number",
        key=f"rad_reg_{scan_id}",
    )

    findings = st.text_area(
        "Findings",
        height=180,
        placeholder=(
            "Describe the radiographic findings..."
        ),
        key=f"findings_{scan_id}",
    )

    impression = st.text_area(
        "Impression",
        height=140,
        placeholder=(
            "Enter the final radiologist impression..."
        ),
        key=f"impression_{scan_id}",
    )

    recommendations = st.text_area(
        "Recommendations",
        height=120,
        placeholder=(
            "Enter recommendations if applicable..."
        ),
        key=f"recommendations_{scan_id}",
    )

    remarks = st.text_area(
        "Radiologist remarks",
        height=100,
        placeholder=(
            "Additional clinical remarks..."
        ),
        key=f"remarks_{scan_id}",
    )

    approval = st.checkbox(
        "I have personally reviewed the medical "
        "image and approve this final interpretation.",
        key=f"approval_{scan_id}",
    )

    # ============================================================
    # APPROVE
    # ============================================================

    if st.button(
        "✓ Approve & Generate Final Medical Report",
        type="primary",
        use_container_width=True,
        key=f"approve_{scan_id}",
    ):

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

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
                "Findings are required."
            )

            return

        if not impression.strip():

            st.error(
                "Impression is required."
            )

            return

        if not approval:

            st.error(
                "The radiologist must explicitly "
                "approve the examination."
            )

            return

        try:

            reviewed_at = datetime.now()

            # ====================================================
            # SAVE REVIEW
            # ====================================================

            review_response = (
                supabase
                .table("radiologist_reviews")
                .insert({
                    "scan_id":
                        scan_id,

                    "user_id":
                        user.id,

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
                    "The review could not be saved."
                )

                return

            review_id = (
                review_response.data[0]["id"]
            )

            # ====================================================
            # REPORT ID
            # ====================================================

            report_id = (
                "MED-R-"
                + datetime.now().strftime("%Y%m%d")
                + "-"
                + uuid.uuid4().hex[:8].upper()
            )

            # ====================================================
            # GENERATE PDF
            # ====================================================

            report_buffer, generated_report_id = (
                generate_pdf_report(

                    patient_name=scan.get(
                        "patient_name",
                        "",
                    ),

                    patient_id=scan.get(
                        "patient_id",
                        "",
                    ),

                    state=scan.get(
                        "patient_state",
                        "",
                    ),

                    examination=scan.get(
                        "examination",
                        "",
                    ),

                    ai_prediction=prediction,

                    ai_confidence=confidence,

                    probabilities=probabilities,

                    radiologist_name=
                        radiologist_name.strip(),

                    registration_number=
                        registration_number.strip(),

                    findings=
                        findings.strip(),

                    impression=
                        impression.strip(),

                    recommendations=
                        recommendations.strip(),

                    remarks=
                        remarks.strip(),

                    reviewed_at=
                        reviewed_at.strftime(
                            "%d %B %Y, %H:%M"
                        ),

                    xray_image=(
                        image_bytes
                        if scan.get(
                            "examination"
                        ) == "Chest X-ray"
                        else None
                    ),

                    ultrasound_image=(
                        image_bytes
                        if scan.get(
                            "examination"
                        ) != "Chest X-ray"
                        else None
                    ),
                )
            )

            pdf_bytes = (
                report_buffer.getvalue()
            )

            # ====================================================
            # SAVE PDF
            # ====================================================

            pdf_path = (
                f"{user.id}/"
                f"{scan.get('patient_id')}/"
                f"{generated_report_id}.pdf"
            )

            supabase.storage.from_(
                "medical-reports"
            ).upload(
                pdf_path,
                pdf_bytes,
                {
                    "content-type":
                        "application/pdf",
                    "upsert":
                        "false",
                },
            )

            # ====================================================
            # REPORT DATABASE
            # ====================================================

            report_response = (
                supabase
                .table("medical_reports")
                .insert({
                    "report_id":
                        generated_report_id,

                    "scan_id":
                        scan_id,

                    "review_id":
                        review_id,

                    "user_id":
                        user.id,

                    "patient_id":
                        scan.get(
                            "patient_id"
                        ),

                    "patient_name":
                        scan.get(
                            "patient_name"
                        ),

                    "patient_state":
                        scan.get(
                            "patient_state"
                        ),

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
                    "Report generated but database "
                    "record could not be created."
                )

                return

            # ====================================================
            # MARK SCAN APPROVED
            # ====================================================

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

            st.success(
                "✅ Radiologist review approved."
            )

            st.success(
                f"✅ Final report generated: "
                f"{generated_report_id}"
            )

            st.download_button(
                "⬇️ Download Approved Report",
                data=pdf_bytes,
                file_name=(
                    f"{generated_report_id}.pdf"
                ),
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                "The examination could not be approved."
            )

            st.exception(error)
