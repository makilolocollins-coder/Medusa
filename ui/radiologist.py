from datetime import datetime

import streamlit as st

from utils.supabase_client import get_supabase
from reports.pdf_report import generate_pdf_report


def get_current_user():

    try:

        supabase = get_supabase()
        response = supabase.auth.get_user()

        if response.user:
            return response.user

    except Exception:
        pass

    return None


def show_radiologist():

    st.title("Radiologist Dashboard")

    st.caption(
        "Clinical review and final report authorization"
    )

    user = get_current_user()

    if user is None:

        st.error(
            "Please log in again."
        )

        return

    supabase = get_supabase()

    # ============================================================
    # VERIFY RADIOLOGIST
    # ============================================================

    try:

        doctors = (
            supabase
            .table("radiologists")
            .select("*")
            .eq(
                "user_id",
                user.id,
            )
            .eq(
                "active",
                True,
            )
            .limit(1)
            .execute()
            .data
            or []
        )

    except Exception as error:

        st.error(
            "Could not verify radiologist account."
        )

        st.exception(error)

        return

    if not doctors:

        st.error(
            "Radiologist access is not enabled "
            "for this account."
        )

        return

    doctor = doctors[0]

    radiologist_name = doctor.get(
        "full_name",
        user.email,
    )

    registration_number = doctor.get(
        "registration_number",
        "",
    )

    st.success(
        f"Signed in as Dr. {radiologist_name}"
    )

    # ============================================================
    # DASHBOARD METRICS
    # ============================================================

    try:

        pending_count = len(
            (
                supabase
                .table("radiologist_requests")
                .select("id")
                .eq("status", "PENDING")
                .execute()
                .data
                or []
            )
        )

        approved_count = len(
            (
                supabase
                .table("radiologist_reviews")
                .select("id")
                .eq(
                    "radiologist_name",
                    radiologist_name,
                )
                .eq(
                    "status",
                    "APPROVED",
                )
                .execute()
                .data
                or []
            )
        )

    except Exception:

        pending_count = 0
        approved_count = 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Pending Reviews",
            pending_count,
        )

    with col2:
        st.metric(
            "Approved Reviews",
            approved_count,
        )

    with col3:
        st.metric(
            "Clinical Status",
            "Active",
        )

    st.divider()

    # ============================================================
    # GET PENDING REQUESTS
    # ============================================================

    try:

        requests = (
            supabase
            .table("radiologist_requests")
            .select("*")
            .eq(
                "status",
                "PENDING",
            )
            .order(
                "created_at",
                desc=True,
            )
            .execute()
            .data
            or []
        )

    except Exception as error:

        st.error(
            "Could not load review requests."
        )

        st.exception(error)

        return

    if not requests:

        st.info(
            "No examinations are currently "
            "waiting for radiologist review."
        )

        return

    st.subheader(
        "Examinations Awaiting Review"
    )

    # ============================================================
    # REQUEST SELECTOR
    # ============================================================

    request_options = []

    for request in requests:

        request_id = request.get("id")
        scan_id = request.get("scan_id")

        request_options.append(
            f"{request_id} | Scan: {scan_id}"
        )

    selected_request = st.selectbox(
        "Select examination",
        request_options,
        key="radiologist_request_selector",
    )

    selected_index = request_options.index(
        selected_request
    )

    request = requests[selected_index]

    scan_id = request.get("scan_id")

    # ============================================================
    # GET SCAN
    # ============================================================

    try:

        scans = (
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

    except Exception as error:

        st.error(
            "Could not load the examination."
        )

        st.exception(error)

        return

    if not scans:

        st.error(
            "The associated scan could not be found."
        )

        return

    scan = scans[0]

    # ============================================================
    # PATIENT / EXAMINATION
    # ============================================================

    st.divider()

    st.subheader(
        "Patient & Examination"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Patient:** "
            f"{scan.get('patient_name', 'Unknown')}"
        )

        st.write(
            f"**Patient ID:** "
            f"{scan.get('patient_id', 'Unknown')}"
        )

        st.write(
            f"**State:** "
            f"{scan.get('patient_state', 'Unknown')}"
        )

    with col2:

        st.write(
            f"**Examination:** "
            f"{scan.get('examination', 'Unknown')}"
        )

        st.write(
            f"**AI Model:** "
            f"{scan.get('model', 'Unknown')}"
        )

        st.write(
            f"**AI Finding:** "
            f"{scan.get('prediction', 'Unknown')}"
        )

        confidence = float(
            scan.get(
                "confidence",
                0,
            )
        )

        st.write(
            f"**AI Confidence:** "
            f"{confidence:.1%}"
        )

    # ============================================================
    # AI PROBABILITIES
    # ============================================================

    probabilities = scan.get(
        "probabilities",
        {},
    )

    if isinstance(
        probabilities,
        dict,
    ) and probabilities:

        st.subheader(
            "AI Probability Distribution"
        )

        for label, value in probabilities.items():

            value = float(value)

            st.write(
                f"{label}: {value:.2%}"
            )

            st.progress(
                min(
                    max(
                        value,
                        0,
                    ),
                    1,
                )
            )

    # ============================================================
    # RADIOLOGIST REVIEW
    # ============================================================

    st.divider()

    st.subheader(
        "Clinical Review"
    )

    st.info(
        "Complete the clinical interpretation below. "
        "The final report will only be generated after "
        "you explicitly approve the examination."
    )

    findings = st.text_area(
        "Findings",
        height=180,
        placeholder=(
            "Describe the imaging findings..."
        ),
        key=f"findings_{scan_id}",
    )

    impression = st.text_area(
        "Impression",
        height=130,
        placeholder=(
            "Enter the final clinical impression..."
        ),
        key=f"impression_{scan_id}",
    )

    recommendations = st.text_area(
        "Recommendations",
        height=110,
        placeholder=(
            "Enter recommendations if applicable..."
        ),
        key=f"recommendations_{scan_id}",
    )

    remarks = st.text_area(
        "Radiologist Remarks",
        height=110,
        placeholder=(
            "Additional professional remarks..."
        ),
        key=f"remarks_{scan_id}",
    )

    approval = st.checkbox(
        "I have personally reviewed this examination "
        "and approve the clinical interpretation.",
        key=f"approval_{scan_id}",
    )

    # ============================================================
    # APPROVE
    # ============================================================

    if st.button(
        "Approve Examination & Generate Final Report",
        type="primary",
        use_container_width=True,
        key=f"approve_{scan_id}",
    ):

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

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
                "You must explicitly approve "
                "the examination."
            )

            return

        try:

            reviewed_at = datetime.now()

            # ====================================================
            # CREATE RADIOLOGIST REVIEW
            #
            # THIS IS THE FIRST PLACE WHERE
            # radiologist_reviews IS INSERTED.
            #
            # Therefore radiologist_name is NEVER NULL.
            # ====================================================

            review_response = (
                supabase
                .table("radiologist_reviews")
                .insert({
                    "scan_id": scan_id,

                    "user_id": user.id,

                    "radiologist_name":
                        str(
                            radiologist_name
                        ).strip(),

                    "registration_number":
                        str(
                            registration_number
                        ).strip(),

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

            # ====================================================
            # UPDATE REQUEST
            # ====================================================

            (
                supabase
                .table("radiologist_requests")
                .update({
                    "status":
                        "APPROVED",
                })
                .eq(
                    "id",
                    request["id"],
                )
                .execute()
            )

            # ====================================================
            # UPDATE SCAN
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

            # ====================================================
            # GENERATE FINAL PDF
            # ====================================================

            report_buffer, report_id = (
                generate_pdf_report(
                    patient_name=(
                        scan.get(
                            "patient_name",
                            "Unknown",
                        )
                    ),

                    patient_id=(
                        scan.get(
                            "patient_id",
                            "Unknown",
                        )
                    ),

                    state=(
                        scan.get(
                            "patient_state",
                            "Unknown",
                        )
                    ),

                    examination=(
                        scan.get(
                            "examination",
                            "Medical Imaging",
                        )
                    ),

                    ai_prediction=(
                        scan.get(
                            "prediction",
                            "Unknown",
                        )
                    ),

                    ai_confidence=(
                        confidence
                    ),

                    probabilities=(
                        probabilities
                    ),

                    radiologist_name=(
                        radiologist_name
                    ),

                    registration_number=(
                        registration_number
                    ),

                    findings=(
                        findings.strip()
                    ),

                    impression=(
                        impression.strip()
                    ),

                    recommendations=(
                        recommendations.strip()
                    ),

                    remarks=(
                        remarks.strip()
                    ),

                    reviewed_at=(
                        reviewed_at.strftime(
                            "%d %B %Y, %H:%M"
                        )
                    ),

                    xray_image=None,

                    ultrasound_image=None,
                )
            )

            pdf_bytes = (
                report_buffer.getvalue()
            )

            # ====================================================
            # SAVE PDF
            # ====================================================

            patient_id = scan.get(
                "patient_id",
                "UNKNOWN",
            )

            pdf_path = (
                f"{user.id}/"
                f"{patient_id}/"
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

            # ====================================================
            # SAVE MEDICAL REPORT
            # ====================================================

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
                            "user_id",
                            user.id,
                        ),

                    "patient_id":
                        patient_id,

                    "patient_name":
                        scan.get(
                            "patient_name",
                            "Unknown",
                        ),

                    "patient_state":
                        scan.get(
                            "patient_state",
                            "Unknown",
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
                    "The report was generated but "
                    "could not be saved."
                )

                return

            # ====================================================
            # SUCCESS
            # ====================================================

            st.success(
                "Examination reviewed and approved."
            )

            st.success(
                f"Final report generated: {report_id}"
            )

            st.download_button(
                "Download Final Medical Report",
                data=pdf_bytes,
                file_name=f"{report_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key=f"doctor_download_{report_id}",
            )

            st.rerun()

        except Exception as error:

            st.error(
                "Could not complete the examination review."
            )

            st.exception(error)
