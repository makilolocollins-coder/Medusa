# ============================================================
# MEDUSA AI
# PATIENT DASHBOARD
# Clinical analytics-style dashboard
# NO CUSTOM HTML
# ============================================================

import streamlit as st
from datetime import datetime

from utils.supabase_client import get_supabase


# ============================================================
# PAGE CONFIGURATION
# ============================================================

def show_dashboard():

    st.title("Patient Dashboard")

    st.caption(
        "Your medical imaging activity, AI screening results, "
        "radiologist reviews and finalized reports."
    )

    supabase = get_supabase()

    # ========================================================
    # CURRENT USER
    # ========================================================

    try:

        auth_response = supabase.auth.get_user()

        if not auth_response.user:

            st.error("Please log in again.")
            return

        user_id = auth_response.user.id

    except Exception:

        st.error("Unable to verify your account.")
        return

    # ========================================================
    # LOAD SCANS
    # ========================================================

    try:

        scans_response = (
            supabase
            .table("ai_scans")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        scans = scans_response.data or []

    except Exception as error:

        st.error("Unable to load your examinations.")
        st.exception(error)
        return

    # ========================================================
    # LOAD REPORTS
    # ========================================================

    try:

        reports_response = (
            supabase
            .table("medical_reports")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        reports = reports_response.data or []

    except Exception:

        reports = []

    # ========================================================
    # LOAD REVIEWS
    # ========================================================

    try:

        reviews_response = (
            supabase
            .table("radiologist_reviews")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        reviews = reviews_response.data or []

    except Exception:

        reviews = []

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    total_scans = len(scans)

    total_reports = len(reports)

    approved_reviews = len([
        review
        for review in reviews
        if str(
            review.get("status", "")
        ).upper()
        == "APPROVED"
    ])

    pending_reviews = max(
        total_scans - approved_reviews,
        0,
    )

    # ========================================================
    # HEADER
    # ========================================================

    st.subheader("Clinical Overview")

    # ========================================================
    # MAIN METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            label="Total Examinations",
            value=total_scans,
        )

    with col2:

        st.metric(
            label="Reviews Completed",
            value=approved_reviews,
        )

    with col3:

        st.metric(
            label="Final Reports",
            value=total_reports,
        )

    with col4:

        st.metric(
            label="Pending Reviews",
            value=pending_reviews,
        )

    # ========================================================
    # REPORT STATUS
    # ========================================================

    st.divider()

    st.subheader("Report Status")

    if pending_reviews > 0:

        st.warning(
            f"{pending_reviews} examination(s) "
            "still require radiologist review."
        )

    else:

        if total_scans > 0:

            st.success(
                "All examinations currently have "
                "completed radiologist review."
            )

        else:

            st.info(
                "No examinations have been recorded yet."
            )

    # ========================================================
    # LATEST EXAMINATION
    # ========================================================

    st.divider()

    st.subheader("Latest Examination")

    if not scans:

        st.info(
            "You have no examinations yet."
        )

        if st.button(
            "Start an AI Examination",
            type="primary",
            use_container_width=True,
        ):

            st.session_state.page = "AI Detection"
            st.rerun()

        return

    latest = scans[0]

    latest_model = latest.get(
        "model",
        "Medical Imaging",
    )

    latest_prediction = latest.get(
        "prediction",
        "Unknown",
    )

    latest_confidence = float(
        latest.get(
            "confidence",
            0,
        )
        or 0
    )

    latest_status = latest.get(
        "status",
        "AI_COMPLETED",
    )

    # ========================================================
    # LATEST METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Examination",
            latest.get(
                "examination",
                latest_model,
            ),
        )

    with col2:

        st.metric(
            "AI Finding",
            str(latest_prediction),
        )

    with col3:

        st.metric(
            "AI Confidence",
            f"{latest_confidence:.1%}",
        )

    with col4:

        if latest_status == "RADIOLOGIST_APPROVED":

            st.metric(
                "Report Status",
                "Finalized",
            )

        else:

            st.metric(
                "Report Status",
                "Pending Review",
            )

    # ========================================================
    # CLINICAL SAFETY MESSAGE
    # ========================================================

    st.info(
        "AI findings are screening results only. "
        "A final medical report becomes available "
        "only after radiologist review and approval."
    )

    # ========================================================
    # EXAMINATION HISTORY
    # ========================================================

    st.divider()

    st.subheader("Examination History")

    history_rows = []

    for scan in scans:

        scan_status = str(
            scan.get(
                "status",
                "AI_COMPLETED",
            )
        ).upper()

        if scan_status == "RADIOLOGIST_APPROVED":

            review_status = "Approved"
            report_status = "Available"

        else:

            review_status = "Pending"
            report_status = "Locked"

        created_at = scan.get(
            "created_at",
            "",
        )

        display_date = created_at

        if created_at:

            try:

                parsed_date = datetime.fromisoformat(
                    created_at.replace(
                        "Z",
                        "+00:00",
                    )
                )

                display_date = parsed_date.strftime(
                    "%d %b %Y"
                )

            except Exception:

                display_date = created_at[:10]

        history_rows.append({
            "Date": display_date,

            "Examination": scan.get(
                "examination",
                "Medical Imaging",
            ),

            "AI Finding": scan.get(
                "prediction",
                "Unknown",
            ),

            "Confidence": (
                f"{float(scan.get('confidence', 0) or 0):.1%}"
            ),

            "Review": review_status,

            "Report": report_status,
        })

    st.dataframe(
        history_rows,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # RECENT FINAL REPORTS
    # ========================================================

    st.divider()

    st.subheader("Final Reports")

    if not reports:

        st.info(
            "No finalized medical reports are available."
        )

    else:

        report_rows = []

        for report in reports:

            report_rows.append({
                "Report ID": report.get(
                    "report_id",
                    "N/A",
                ),

                "Patient ID": report.get(
                    "patient_id",
                    "N/A",
                ),

                "Status": report.get(
                    "status",
                    "N/A",
                ),

                "Approved": report.get(
                    "approved_at",
                    "N/A",
                ),
            })

        st.dataframe(
            report_rows,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # QUICK ACTIONS
    # ========================================================

    st.divider()

    st.subheader("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "🔬 New Examination",
            use_container_width=True,
        ):

            st.session_state.page = "AI Detection"
            st.rerun()

    with col2:

        if st.button(
            "📄 View Reports",
            use_container_width=True,
        ):

            st.session_state.page = "Reports"
            st.rerun()

    with col3:

        if st.button(
            "📞 Consultation",
            use_container_width=True,
        ):

            st.session_state.page = "Health"
            st.rerun()

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "Medusa AI provides AI-assisted screening information "
        "and does not replace professional medical diagnosis, "
        "medical advice, or treatment."
    )
