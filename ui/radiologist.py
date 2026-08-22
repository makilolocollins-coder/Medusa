# ================================================================
# MEDUSA AI
# RADIOLOGIST PORTAL
#
# ONLY RADIOLOGISTS CAN:
# - View pending examinations
# - Review images
# - Enter findings
# - Enter impression
# - Enter recommendations
# - Add remarks
# - Approve examination
# - Generate final medical report
#
# PATIENTS CANNOT COMPLETE THIS FORM.
# ================================================================

import io
from datetime import datetime

import streamlit as st
from PIL import Image

from utils.supabase_client import get_supabase
from reports.pdf_report import generate_pdf_report


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
# RADIOLOGIST AUTHORIZATION
# ================================================================

def get_radiologist():

    user = get_current_user()

    if user is None:
        return None

    supabase = get_supabase()

    doctors = (
        supabase
        .table("radiologists")
        .select(
            "user_id,full_name,active"
        )
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

    if not doctors:
        return None

    return doctors[0]


# ================================================================
# LOAD PENDING REQUESTS
# ================================================================

def load_pending_requests():

    supabase = get_supabase()

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
            desc=False,
        )
        .execute()
        .data
        or []
    )

    return requests


# ================================================================
# LOAD SCAN
# ================================================================

def load_scan(scan_id):

    supabase = get_supabase()

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

    if not scans:
        return None

    return scans[0]


# ================================================================
# LOAD EXISTING REVIEW
# ================================================================

def load_review(scan_id):

    supabase = get_supabase()

    reviews = (
        supabase
        .table("radiologist_reviews")
        .select("*")
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

    if not reviews:
        return None

    return reviews[0]


# ================================================================
# MAIN
# ================================================================

def show_radiologist():

    st.title(
        "👨‍⚕️ Radiologist Portal"
    )

    st.caption(
        "Medusa AI Clinical Review Workspace"
    )

    # ============================================================
    # AUTHORIZATION
    # ============================================================

    radiologist = get_radiologist()

    if radiologist is None:

        st.error(
            "Access denied."
        )

        st.warning(
            "This area is restricted to "
            "authorized radiologists."
        )

        st.stop()

    radiologist_name = (
        radiologist.get("full_name")
        or "Radiologist"
    )

    st.success(
        f"Signed in as {radiologist_name}"
    )

    # ============================================================
    # DASHBOARD METRICS
    # ============================================================

    try:

        requests = load_pending_requests()

        pending_count = len(requests)

        supabase = get_supabase()

        approved_reviews = (
            supabase
            .table("radiologist_reviews")
            .select(
                "id",
                count="exact",
            )
            .eq(
                "status",
                "APPROVED",
            )
            .execute()
        )

        approved_count = (
            approved_reviews.count
            if approved_reviews.count is not None
            else 0
        )

        total_reviews = (
            supabase
            .table("radiologist_reviews")
            .select(
                "id",
                count="exact",
            )
            .execute()
        )

        total_count = (
            total_reviews.count
            if total_reviews.count is not None
            else 0
        )

    except Exception as error:

        st.error(
            "Unable to load radiologist dashboard."
        )

        st.exception(error)

        return

    st.subheader(
        "Clinical Review Dashboard"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Pending Reviews",
            pending_count,
        )

    with c2:

        st.metric(
            "Completed Reviews",
            approved_count,
        )

    with c3:

        st.metric(
            "Total Reviews",
            total_count,
        )

    # ============================================================
    # PENDING EXAMINATIONS
    # ============================================================

    st.divider()

    st.subheader(
        "Pending Examinations"
    )

    if not requests:

        st.success(
            "No examinations are currently "
            "waiting for review."
        )

        return

    # ============================================================
    # REQUEST SELECTOR
    # ============================================================

    request_options = {}

    for request in requests:

        scan_id = request.get(
            "scan_id"
        )

        if not scan_id:
            continue

        scan = load_scan(scan_id)

        if not scan:
            continue

        patient_id = (
            scan.get("patient_id")
            or "Unknown"
        )

        patient_name = (
            scan.get("patient_name")
            or "Unknown patient"
        )

        examination = (
            scan.get("examination")
            or "Medical imaging"
        )

        label = (
            f"{patient_name} | "
            f"{patient_id} | "
            f"{examination}"
        )

        request_options[label] = (
            request,
            scan,
        )

    if not request_options:

        st.warning(
            "No valid examination records "
            "were found."
        )

        return

    selected_label = st.selectbox(
        "Select examination to review",
        list(request_options.keys()),
        key="radiologist_selected_request",
    )

    selected_request, scan = (
        request_options[selected_label]
    )

    scan_id = scan.get("id")

    # ============================================================
    # PATIENT / EXAMINATION DETAILS
    # ============================================================

    st.divider()

    st.subheader(
        "Examination Details"
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

    st.write(
        f"**AI Prediction:** "
        f"{scan.get('prediction', 'Unknown')}"
    )

    confidence = scan.get(
        "confidence"
    )

    if confidence is not None:

        try:

            st.write(
                f"**AI Confidence:** "
                f"{float(confidence):.2%}"
            )

        except Exception:
            pass

    # ============================================================
    # MEDICAL IMAGE
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

            supabase = get_supabase()

            image_bytes = (
                supabase.storage
                .from_("mammosense-scans")
                .download(image_path)
            )

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            st.image(
                image,
                caption=(
                    scan.get(
                        "examination",
                        "Medical image",
                    )
                ),
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                "Unable to load the examination image."
            )

            st.exception(error)

    else:

        st.warning(
            "No image path is associated "
            "with this examination."
        )

    # ============================================================
    # AI PROBABILITIES
    # ============================================================

    probabilities = scan.get(
        "probabilities",
        {},
    )

    if (
        isinstance(probabilities, dict)
        and probabilities
    ):

        st.subheader(
            "AI Probability Breakdown"
        )

        for name, value in probabilities.items():

            try:

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

            except Exception:
                continue

    # ============================================================
    # EXISTING REVIEW
    # ============================================================

    existing_review = load_review(
        scan_id
    )

    if existing_review:

        existing_status = str(
            existing_review.get(
                "status",
                "",
            )
        ).upper()

        if existing_status == "APPROVED":

            st.success(
                "This examination has already "
                "been approved."
            )

            st.write(
                f"**Reviewed by:** "
                f"{existing_review.get('radiologist_name', '')}"
            )

            st.write(
                f"**Impression:** "
                f"{existing_review.get('impression', '')}"
            )

            return

    # ============================================================
    # RADIOLOGIST REVIEW
    # ============================================================

    st.divider()

    st.subheader(
        "Clinical Review"
    )

    st.info(
        "Complete the clinical review before "
        "approving this examination."
    )

    review_name = st.text_input(
        "Radiologist name",
        value=radiologist_name,
        key=f"review_name_{scan_id}",
    )

    registration_number = st.text_input(
        "Medical registration number",
        key=f"registration_{scan_id}",
    )

    findings = st.text_area(
        "Findings",
        placeholder=(
            "Describe the radiological findings..."
        ),
        height=180,
        key=f"findings_{scan_id}",
    )

    impression = st.text_area(
        "Impression",
        placeholder=(
            "Enter the final radiological impression..."
        ),
        height=140,
        key=f"impression_{scan_id}",
    )

    recommendations = st.text_area(
        "Recommendations",
        placeholder=(
            "Enter recommendations if applicable..."
        ),
        height=120,
        key=f"recommendations_{scan_id}",
    )

    remarks = st.text_area(
        "Radiologist remarks",
        placeholder=(
            "Additional professional remarks..."
        ),
        height=120,
        key=f"remarks_{scan_id}",
    )

    approval = st.checkbox(
        "I confirm that I have personally reviewed "
        "the examination and approve the clinical "
        "interpretation entered above.",
        key=f"approval_{scan_id}",
    )

    # ============================================================
    # APPROVE
    # ============================================================

    if st.button(
        "✓ Approve Review & Generate Final Report",
        type="primary",
        use_container_width=True,
        key=f"approve_{scan_id}",
    ):

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        if not review_name.strip():

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
                "Impression is required."
            )

            return

        if not approval:

            st.error(
                "You must explicitly approve "
                "the examination."
            )

            return

        # --------------------------------------------------------
        # CURRENT USER
        # --------------------------------------------------------

        user = get_current_user()

        if user is None:

            st.error(
                "Your session has expired."
            )

            return

        try:

            supabase = get_supabase()

            reviewed_at = datetime.now()

            # ====================================================
            # SAVE RADIOLOGIST REVIEW
            # ====================================================

            review_response = (
                supabase
                .table("radiologist_reviews")
                .insert({
                    "scan_id": scan_id,
                    "user_id": user.id,
                    "radiologist_name":
                        review_name.strip(),
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
                    "Radiologist review could "
                    "not be saved."
                )

                return

            review_id = (
                review_response.data[0]["id"]
            )

            # ====================================================
            # UPDATE REQUEST
            # ====================================================

            request_id = (
                selected_request.get("id")
            )

            if request_id:

                (
                    supabase
                    .table("radiologist_requests")
                    .update({
                        "status": "APPROVED",
                    })
                    .eq(
                        "id",
                        request_id,
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

            probabilities = scan.get(
                "probabilities",
                {},
            )

            report_buffer, report_id = (
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
                    examination=(
                        scan.get(
                            "examination",
                            "",
                        )
                    ),
                    ai_prediction=(
                        scan.get(
                            "prediction",
                            "",
                        )
                    ),
                    ai_confidence=float(
                        scan.get(
                            "confidence",
                            0,
                        )
                    ),
                    probabilities=(
                        probabilities
                    ),
                    radiologist_name=(
                        review_name.strip()
                    ),
                    registration_number=(
                        registration_number.strip()
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
                    xray_image=(
                        image_bytes
                        if (
                            scan.get(
                                "examination"
                            )
                            == "Chest X-ray"
                        )
                        else None
                    ),
                    ultrasound_image=(
                        image_bytes
                        if (
                            scan.get(
                                "examination"
                            )
                            == "Breast Ultrasound"
                        )
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

            patient_id = scan.get(
                "patient_id"
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
            # SAVE REPORT
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
                            "user_id"
                        ),
                    "patient_id":
                        patient_id,
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
                    "The report was generated but "
                    "could not be registered."
                )

                return

            # ====================================================
            # SUCCESS
            # ====================================================

            st.success(
                "✅ Radiologist review approved."
            )

            st.success(
                f"✅ Final medical report generated: "
                f"{report_id}"
            )

            st.download_button(
                label=(
                    "⬇️ Download Final Medical Report"
                ),
                data=pdf_bytes,
                file_name=(
                    f"{report_id}.pdf"
                ),
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key=f"radiologist_download_{report_id}",
            )

            st.info(
                "The patient can now access the "
                "approved final report."
            )

            st.rerun()

        except Exception as error:

            st.error(
                "Could not complete the radiologist review."
            )

            st.exception(error)

    # ============================================================
    # SECURITY NOTICE
    # ============================================================

    st.divider()

    st.caption(
        "Medusa AI Clinical Review Portal • "
        "Authorized radiologists only"
    )
