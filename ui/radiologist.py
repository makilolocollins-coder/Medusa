import streamlit as st

from ui.background import set_background
from utils.supabase_client import get_supabase


def show_radiologist():

    set_background("radiologist.jpg")

    st.title("👨‍⚕️ Radiologist Portal")

    st.caption(
        "MammoSense professional review workspace"
    )

    supabase = get_supabase()

    # ========================================================
    # AUTHENTICATION
    # ========================================================

    user_response = supabase.auth.get_user()

    if not user_response.user:

        st.error(
            "You must be logged in."
        )

        return

    radiologist_id = user_response.user.id

    # ========================================================
    # CHECK RADIOLOGIST
    # ========================================================

    try:

        doctor = (
            supabase
            .table("radiologists")
            .select("*")
            .eq(
                "user_id",
                radiologist_id
            )
            .eq(
                "active",
                True
            )
            .execute()
            .data
        )

    except Exception as error:

        st.error(
            "Unable to verify radiologist account."
        )

        st.exception(error)

        return

    if not doctor:

        st.error(
            "This account is not authorized "
            "as a radiologist."
        )

        return

    # ========================================================
    # LOAD PENDING REQUESTS
    # ========================================================

    try:

        requests = (
            supabase
            .table("radiologist_requests")
            .select(
                "id, user_id, scan_id, status, "
                "created_at"
            )
            .eq(
                "status",
                "Pending"
            )
            .order(
                "created_at",
                desc=True
            )
            .execute()
            .data
            or []
        )

    except Exception as error:

        st.error(
            "Unable to load consultation requests."
        )

        st.exception(error)

        return

    # ========================================================
    # DASHBOARD
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Pending Reviews",
            len(requests)
        )

    with col2:

        st.metric(
            "Status",
            "Online"
        )

    st.divider()

    if not requests:

        st.success(
            "No pending radiologist reviews."
        )

        return

    # ========================================================
    # REQUESTS
    # ========================================================

    st.subheader(
        "Pending Reviews"
    )

    for request in requests:

        with st.container(border=True):

            st.write(
                f"### Review Request"
            )

            st.caption(
                f"Request ID: {request['id']}"
            )

            st.caption(
                f"Submitted: "
                f"{request['created_at']}"
            )

            if st.button(
                "Open Scan",
                key=f"open_{request['id']}",
                use_container_width=True
            ):

                st.session_state[
                    "selected_request"
                ] = request

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

    st.subheader(
        "🔍 Scan Review"
    )

    scan_id = selected["scan_id"]

    # ========================================================
    # LOAD SCAN
    # ========================================================

    try:

        scan_response = (
            supabase
            .table("ai_scans")
            .select("*")
            .eq(
                "id",
                scan_id
            )
            .single()
            .execute()
        )

        scan = scan_response.data

    except Exception as error:

        st.error(
            "Unable to load scan."
        )

        st.exception(error)

        return

    # ========================================================
    # AI RESULT
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "AI Finding",
            scan.get(
                "prediction",
                "Unknown"
            )
        )

    with col2:

        confidence = scan.get(
            "confidence",
            0
        )

        st.metric(
            "AI Confidence",
            f"{confidence:.1%}"
        )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    probabilities = scan.get(
        "probabilities",
        {}
    )

    if probabilities:

        st.subheader(
            "AI Probability Breakdown"
        )

        for name, value in probabilities.items():

            st.write(
                f"**{name}: {value:.1%}**"
            )

            st.progress(
                min(max(value, 0), 1)
            )

    # ========================================================
    # RADIOLOGIST NOTE
    # ========================================================

    st.subheader(
        "Professional Review"
    )

    note = st.text_area(
        "Radiologist note",
        placeholder=(
            "Enter your professional observations "
            "and review..."
        ),
        height=180,
        key=f"note_{scan_id}"
    )

    # ========================================================
    # MARK REVIEWED
    # ========================================================

    if st.button(
        "✓ Mark as Reviewed",
        type="primary",
        use_container_width=True
    ):

        if not note.strip():

            st.warning(
                "Please enter a review note."
            )

            return

        try:

            (
                supabase
                .table("radiologist_requests")
                .update({
                    "status": "Reviewed",
                    "radiologist_id": radiologist_id,
                    "radiologist_note": note,
                    "reviewed_at": "now()"
                })
                .eq(
                    "id",
                    selected["id"]
                )
                .execute()
            )

            st.success(
                "Review completed successfully."
            )

            del st.session_state[
                "selected_request"
            ]

            st.rerun()

        except Exception as error:

            st.error(
                "Unable to complete review."
            )

            st.exception(error)
