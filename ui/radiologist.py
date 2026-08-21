import streamlit as st

from ui.background import set_background
from utils.supabase_client import get_supabase


def show_radiologist():

    set_background("radiologist.jpg")

    st.title("👨‍⚕️ Radiologist Portal")
    st.caption("MammoSense professional review")

    supabase = get_supabase()

    # ========================================================
    # CURRENT USER
    # ========================================================

    response = supabase.auth.get_user()

    if not response.user:
        st.error("Please log in.")
        return

    radiologist_id = response.user.id

    # ========================================================
    # VERIFY RADIOLOGIST
    # ========================================================

    doctor = (
        supabase
        .table("radiologists")
        .select("user_id, full_name, active")
        .eq("user_id", radiologist_id)
        .eq("active", True)
        .execute()
        .data
        or []
    )

    if not doctor:
        st.error(
            "This account is not authorized as a radiologist."
        )
        return

    doctor_name = doctor[0].get(
        "full_name",
        "Radiologist"
    )

    st.success(
        f"Welcome, {doctor_name}"
    )

    # ========================================================
    # GET REQUESTS
    # ========================================================

    try:

        requests = (
            supabase
            .table("radiologist_requests")
            .select("*")
            .order("created_at", desc=True)
            .execute()
            .data
            or []
        )

    except Exception as error:

        st.error(
            f"Could not load review requests: {error}"
        )

        return

    # ========================================================
    # FILTER PENDING
    # ========================================================

    pending_requests = [
        request
        for request in requests
        if str(
            request.get("status", "")
        ).lower() == "pending"
    ]

    reviewed_requests = [
        request
        for request in requests
        if str(
            request.get("status", "")
        ).lower() == "reviewed"
    ]

    # ========================================================
    # DASHBOARD METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pending Reviews",
            len(pending_requests),
        )

    with col2:

        st.metric(
            "Reviewed",
            len(reviewed_requests),
        )

    with col3:

        st.metric(
            "Total Requests",
            len(requests),
        )

    st.divider()

    # ========================================================
    # NO REQUESTS
    # ========================================================

    if not pending_requests:

        st.success(
            "No pending radiologist reviews."
        )

        return

    # ========================================================
    # PENDING REVIEWS
    # ========================================================

    st.subheader("📋 Pending Reviews")

    for request in pending_requests:

        request_id = request.get("id")
        scan_id = request.get("scan_id")
        created_at = request.get("created_at")

        # ----------------------------------------------------
        # GET SCAN
        # ----------------------------------------------------

        try:

            scan_response = (
                supabase
                .table("ai_scans")
                .select("*")
                .eq("id", scan_id)
                .single()
                .execute()
            )

            scan = scan_response.data

        except Exception:

            scan = None

        # ----------------------------------------------------
        # CARD
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown(
                "### 🧬 MammoSense Review"
            )

            if scan:

                prediction = scan.get(
                    "prediction",
                    "Unknown",
                )

                confidence = scan.get(
                    "confidence",
                    0,
                )

                st.write(
                    f"**AI Finding:** {prediction}"
                )

                st.write(
                    f"**AI Confidence:** "
                    f"{confidence:.1%}"
                )

            else:

                st.warning(
                    "The associated scan could not be loaded."
                )

            st.caption(
                f"Submitted: {created_at}"
            )

            if st.button(
                "Open Scan",
                key=f"open_{request_id}",
                use_container_width=True,
            ):

                st.session_state.selected_request = request

                st.rerun()

    # ========================================================
    # SELECTED REQUEST
    # ========================================================

    selected = st.session_state.get(
        "selected_request"
    )

    if not selected:
        return

    st.divider()

    st.subheader("🔎 Scan Review")

    selected_scan_id = selected.get(
        "scan_id"
    )

    try:

        scan = (
            supabase
            .table("ai_scans")
            .select("*")
            .eq("id", selected_scan_id)
            .single()
            .execute()
            .data
        )

    except Exception as error:

        st.error(
            f"Could not load scan: {error}"
        )

        return

    if not scan:

        st.error(
            "Scan information was not found."
        )

        return

    # ========================================================
    # RESULTS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "AI Finding",
            scan.get(
                "prediction",
                "Unknown",
            ),
        )

    with col2:

        confidence = scan.get(
            "confidence",
            0,
        )

        st.metric(
            "AI Confidence",
            f"{confidence:.1%}",
        )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader(
        "AI Probability Breakdown"
    )

    probabilities = scan.get(
        "probabilities",
        {},
    )

    if isinstance(probabilities, dict):

        for name, value in probabilities.items():

            st.write(
                f"**{name}** — {value:.1%}"
            )

            st.progress(
                float(value)
            )

    # ========================================================
    # RADIOLOGIST NOTE
    # ========================================================

    st.subheader(
        "Radiologist Review"
    )

    note = st.text_area(
        "Professional review note",
        height=180,
        placeholder=(
            "Enter your professional interpretation "
            "and recommendations..."
        ),
        key=f"note_{selected['id']}",
    )

    # ========================================================
    # MARK REVIEWED
    # ========================================================

    if st.button(
        "✓ Mark as Reviewed",
        type="primary",
        use_container_width=True,
    ):

        if not note.strip():

            st.warning(
                "Please enter a review note first."
            )

            return

        try:

            (
                supabase
                .table("radiologist_requests")
                .update(
                    {
                        "status": "Reviewed",
                        "radiologist_id": radiologist_id,
                        "radiologist_note": note.strip(),
                    }
                )
                .eq(
                    "id",
                    selected["id"],
                )
                .execute()
            )

            st.success(
                "Radiologist review completed."
            )

            st.session_state.pop(
                "selected_request",
                None,
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Could not save review: {error}"
        )
