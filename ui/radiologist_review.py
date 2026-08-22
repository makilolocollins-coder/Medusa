# ================================================================
# MEDUSA AI
# RADIOLOGIST REVIEW WORKSPACE
#
# ONLY RADIOLOGIST USERS SHOULD HAVE ACCESS TO THIS PAGE.
#
# WORKFLOW:
#
# PENDING_REVIEW
#       ↓
# Radiologist reviews image
#       ↓
# Radiologist enters findings
#       ↓
# Radiologist approves
#       ↓
# RADIOLOGIST_APPROVED
#       ↓
# PDF GENERATED
#       ↓
# PDF AVAILABLE FOR DOWNLOAD
# ================================================================

import io
from datetime import datetime

import streamlit as st
from PIL import Image

from utils.supabase_client import get_supabase

from reports.pdf_report import (
    generate_pdf_report,
)


# ================================================================
# CONSTANTS
# ================================================================

REVIEW_PENDING = "PENDING_REVIEW"
REVIEW_APPROVED = "RADIOLOGIST_APPROVED"


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
# LOAD PENDING SCANS
# ================================================================

def get_pending_scans():

    supabase = get_supabase()

    response = (
        supabase
        .table("ai_scans")
        .select("*")
        .eq(
            "status",
            REVIEW_PENDING,
        )
        .order(
            "created_at",
            desc=False,
        )
        .execute()
    )

    return response.data or []


# ================================================================
# DOWNLOAD IMAGE FROM STORAGE
# ================================================================

def download_scan_image(
    image_path,
):

    supabase = get_supabase()

    image_bytes = (
        supabase
        .storage
        .from_("mammosense-scans")
        .download(image_path)
    )

    return image_bytes


# ================================================================
# MAIN
# ================================================================

def show_radiologist_review():

    st.title(
        "👨‍⚕️ Radiologist Review"
    )

    st.caption(
        "Medusa AI clinical review workspace"
    )

    # ============================================================
    # USER
    # ============================================================

    user = get_current_user()

    if user is None:

        st.error(
            "Please log in."
        )

        return

    # ============================================================
    # IMPORTANT
    #
    # You should additionally protect this page
    # using your Supabase role/profile system.
    # ============================================================

    st.info(
        "Only authorized radiologists should "
        "have access to this workspace."
    )

    # ============================================================
    # PENDING SCANS
    # ============================================================

    try:

        scans = get_pending_scans()

    except Exception as error:

        st.error(
            "Could not load pending examinations."
        )

        st.exception(error)

        return

    if not scans:

        st.success(
            "✓ No examinations are currently "
            "waiting for radiologist review."
        )

        return

    st.metric(
        "Pending examinations",
        len(scans),
    )

    # ============================================================
    # SELECT SCAN
    # ============================================================

    scan_options = {}

    for scan in scans:

        scan_id = scan.get("id")

        patient_name = (
            scan.get(
                "patient_name",
                "Unknown patient",
            )
        )

        examination = (
            scan.get(
                "examination",
                "Examination",
            )
        )

        patient_id = (
            scan.get(
                "patient_id",
                "",
            )
        )

        label = (
            f"{patient_name} | "
            f"{examination} | "
            f"{patient_id}"
        )

        scan_options[label] = scan

    selected_label = st.selectbox(
        "Select examination",
        list(scan_options.keys()),
    )

    scan = scan_options[
        selected_label
    ]

    scan_id = scan["id"]

    # ============================================================
    # PATIENT INFORMATION
    # ============================================================

    st.divider()

    st.subheader(
        "Patient Information"
    )

    patient_col1, patient_col2 = (
        st.columns(2)
    )

    with patient_col1:

        st.write(
            f"**Patient:** "
            f"{scan.get('patient_name', '')}"
        )

        st.write(
            f"**Patient ID:** "
            f"{scan.get('patient_id', '')}"
        )

    with patient_col2:

        st.write(
            f"**State:** "
            f"{scan.get('patient_state', '')}"
        )

        st.write(
            f"**Examination:** "
            f"{scan.get('examination', '')}"
        )

    # ============================================================
    # IMAGE
    # ============================================================

    st.divider()

    st.subheader(
        "Medical Image"
    )

    try:

        image_bytes = download_scan_image(
            scan["image_path"]
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
            "Unable to retrieve the examination image."
        )

        st.exception(error)

        return

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

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "AI Finding",
            prediction,
        )

    with col2:

        st.metric(
            "AI Confidence",
            f"{confidence:.1%}",
        )

    probabilities = scan.get(
        "probabilities",
        {},
    )

    if isinstance(
        probabilities,
        dict,
    ):

        st.write(
            "### AI Probability Breakdown"
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

    st.warning(
        "The AI result is an aid to clinical "
        "assessment. The radiologist must "
        "independently review the medical image."
    )

    # ============================================================
    # RADIOLOGIST
    # ============================================================

    st.divider()

    st.subheader(
        "Radiologist Interpretation"
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
        "Findings *",
        placeholder=(
            "Describe the relevant imaging findings."
        ),
        height=180,
        key=f"rad_findings_{scan_id}",
    )

    impression = st.text_area(
        "Final Impression *",
        placeholder=(
            "Provide the radiologist's final impression."
        ),
        height=140,
        key=f"rad_impression_{scan_id}",
    )

    recommendations = st.text_area(
        "Recommendations",
        placeholder=(
            "Further imaging, clinical correlation, "
            "follow-up or other recommendations."
        ),
        height=110,
        key=f"rad_recommendations_{scan_id}",
    )

    remarks = st.text_area(
        "Radiologist Remarks",
        placeholder=(
            "Additional professional remarks."
        ),
        height=110,
        key=f"rad_remarks_{scan_id}",
    )

    # ============================================================
    # APPROVAL
    # ============================================================

    approval = st.checkbox(
        "I confirm that I have personally reviewed "
        "the medical image and approve the clinical "
        "interpretation entered above.",
        key=f"rad_approval_{scan_id}",
    )

    # ============================================================
    # FINAL APPROVAL
    # ============================================================

    if st.button(
        "✓ Approve Examination & Generate Final Report",
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
                "Medical registration number "
                "is required."
            )

            return

        if not findings.strip():

            st.error(
                "Findings are required."
            )

            return

        if not impression.strip():

            st.error(
                "Final impression is required."
            )

            return

        if not approval:

            st.error(
                "You must explicitly confirm "
                "radiologist approval."
            )

            return

        # --------------------------------------------------------
        # DATABASE
        # --------------------------------------------------------

        try:

            supabase = get_supabase()

            # ----------------------------------------------------
            # RECHECK DATABASE STATUS
            #
            # Prevents approving an already approved scan.
            # ----------------------------------------------------

            latest = (
                supabase
                .table("ai_scans")
                .select("*")
                .eq(
                    "id",
                    scan_id,
                )
                .limit(1)
                .execute()
                .data
                or []
            )

            if not latest:

                st.error(
                    "This examination no longer exists."
                )

                return

            latest_scan = latest[0]

            if latest_scan.get("status") != (
                REVIEW_PENDING
            ):

                st.error(
                    "This examination has already "
                    "been reviewed or its status "
                    "has changed."
                )

                st.rerun()

                return

            # ----------------------------------------------------
            # REVIEW TIME
            # ----------------------------------------------------

            reviewed_at = datetime.now()

            # ----------------------------------------------------
            # SAVE RADIOLOGIST REVIEW
            # ----------------------------------------------------

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
                        REVIEW_APPROVED,

                    "approved":
                        True,

                    "reviewed_at":
                        reviewed_at.isoformat(),
                })
                .execute()
            )

            if not review_response.data:

                st.error(
                    "Radiologist review could not "
                    "be saved."
                )

                return

            review_id = (
                review_response.data[0]["id"]
            )

            # ----------------------------------------------------
            # UPDATE SCAN STATUS
            # ----------------------------------------------------

            update_response = (
                supabase
                .table("ai_scans")
                .update({
                    "status":
                        REVIEW_APPROVED,
                })
                .eq(
                    "id",
                    scan_id,
                )
                .eq(
                    "status",
                    REVIEW_PENDING,
                )
                .execute()
            )

            if not update_response.data:

                st.error(
                    "The scan could not be marked "
                    "as radiologist approved."
                )

                return

            # ----------------------------------------------------
            # GENERATE FINAL PDF
            # ----------------------------------------------------

            examination = scan.get(
                "examination",
                "Medical Imaging",
            )

            is_xray = (
                "x-ray"
                in examination.lower()
            )

            pdf_buffer, report_id = (
                generate_pdf_report(
                    patient_name=(
                        scan.get(
                            "patient_name",
                            "",
                        )
                    ),

                    patient_id=(
                        scan.get(
                            "patient_id",
                            "",
                        )
                    ),

                    state=(
                        scan.get(
                            "patient_state",
                            "",
                        )
                    ),

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
                        if is_xray
                        else None
                    ),

                    ultrasound_image=(
                        image_bytes
                        if not is_xray
                        else None
                    ),
                )
            )

            pdf_bytes = (
                pdf_buffer.getvalue()
            )

            # ----------------------------------------------------
            # SAVE PDF
            # ----------------------------------------------------

            pdf_path = (
                f"{scan.get('user_id')}/"
                f"{scan.get('patient_id')}/"
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

                    "upsert":
                        "false",
                },
            )

            # ----------------------------------------------------
            # SAVE REPORT RECORD
            # ----------------------------------------------------

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
                        scan.get(
                            "user_id"
                        ),

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
                        REVIEW_APPROVED,

                    "pdf_path":
                        pdf_path,

                    "approved_at":
                        reviewed_at.isoformat(),
                })
                .execute()
            )

            if not report_response.data:

                st.error(
                    "The report PDF was generated "
                    "but its database record could "
                    "not be created."
                )

                return

            # ----------------------------------------------------
            # SUCCESS
            # ----------------------------------------------------

            st.success(
                "✅ Examination reviewed and approved."
            )

            st.success(
                "✅ Final medical report generated."
            )

            st.success(
                f"Report ID: {report_id}"
            )

            # ----------------------------------------------------
            # DOWNLOAD
            #
            # This button exists ONLY after successful
            # radiologist approval and report creation.
            # ----------------------------------------------------

            st.download_button(
                label=(
                    "⬇️ Download Approved Medical Report"
                ),

                data=pdf_bytes,

                file_name=(
                    f"{report_id}.pdf"
                ),

                mime="application/pdf",

                type="primary",

                use_container_width=True,

                key=f"download_{report_id}",
            )

        except Exception as error:

            st.error(
                "The radiologist approval process "
                "could not be completed."
            )

            st.exception(error)
