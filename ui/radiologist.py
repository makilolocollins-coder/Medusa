import streamlit as st

from ui.background import set_background
from utils.supabase_client import get_supabase


def show_radiologist():

    set_background("radiologist.jpg")

    st.title("👨‍⚕️ Radiologist Portal")
    st.caption("MammoSense professional review")

    supabase = get_supabase()

    user = supabase.auth.get_user()

    if not user.user:
        st.error("Please log in.")
        return

    radiologist_id = user.user.id

    # Check authorization
    doctor = (
        supabase
        .table("radiologists")
        .select("*")
        .eq("user_id", radiologist_id)
        .eq("active", True)
        .execute()
        .data
    )

    if not doctor:
        st.error(
            "This account is not authorized "
            "as a radiologist."
        )
        return

    # Get pending requests
    requests = (
        supabase
        .table("radiologist_requests")
        .select("*")
        .eq("status", "Pending")
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )

    st.metric(
        "Pending Reviews",
        len(requests)
    )

    st.divider()

    if not requests:
        st.success(
            "No pending radiologist reviews."
        )
        return

    st.subheader("Pending Reviews")

    for request in requests:

        with st.container(border=True):

            st.write(
                "🧬 MammoSense Scan"
            )

            st.caption(
                f"Submitted: "
                f"{request['created_at']}"
            )

            if st.button(
                "Open Scan",
                key=f"open_{request['id']}"
            ):

                st.session_state.selected_request = request
                st.rerun()

    # Selected scan
    selected = st.session_state.get(
        "selected_request"
    )

    if not selected:
        return

    st.divider()

    st.subheader("Scan Review")

    scan = (
        supabase
        .table("ai_scans")
        .select("*")
        .eq("id", selected["scan_id"])
        .single()
        .execute()
        .data
    )

    st.metric(
        "AI Finding",
        scan["prediction"]
    )

    st.metric(
        "AI Confidence",
        f"{scan['confidence']:.1%}"
    )

    st.subheader(
        "AI Probabilities"
    )

    for name, value in scan[
        "probabilities"
    ].items():

        st.write(
            f"{name}: {value:.1%}"
        )

        st.progress(value)

    note = st.text_area(
        "Radiologist Review",
        height=150,
        placeholder="Enter your professional review..."
    )

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

        supabase.table(
            "radiologist_requests"
        ).update({
            "status": "Reviewed",
            "radiologist_id": radiologist_id,
            "radiologist_note": note,
            "reviewed_at": "now()"
        }).eq(
            "id",
            selected["id"]
        ).execute()

        st.success(
            "Review completed."
        )

        del st.session_state.selected_request

        st.rerun()
