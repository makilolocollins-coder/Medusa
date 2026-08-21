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
    # DASHBOARD
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
    # PENDING REVIEWS
    # ========================================================

    st.subheader("📋 Pending Reviews")

    if not pending_requests:

        st.success(
            "No pending radiologist reviews."
        )

    else:

        for request in pending_requests:

            request_id = request.get("id")
            scan_id = request.get("scan_id")

            # ------------------------------------------------
            # GET SCAN INFORMATION
            # ------------------------------------------------

            try:

                scan = (
                    supabase
                    .table("ai_scans")
                    .select("*")
                    .eq("id", scan_id)
                    .single()
                    .execute()
                    .data
                )

            except Exception:

                scan = None

            # ------------------------------------------------
            # REQUEST CARD
            # ------------------------------------------------

            with st.container(border=True):

                st.markdown(
                    "### 🧬 MammoSense Scan"
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

                st.caption(
                    f"Submitted: "
                    f"{request.get('created_at', 'Unknown')}"
                )

                if st.button(
                    "🔎 Open Scan",
                    key=f"open_{request_id}",
                    use_container_width=True,
                ):

                    st.session_state.selected_request = request

                    st.rerun()

    # ========================================================
    # SELECTED SCAN
    # ========================================================

    selected = st.session_state.get(
        "selected_request"
    )

    if not selected:
        return

    st.divider()

    st.subheader("🔬 Scan Review")

    selected_scan_id = selected.get(
        "scan_id"
    )

    # ========================================================
    # LOAD SCAN
    # ========================================================

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
    # LOAD ACTUAL ULTRASOUND
    # ========================================================

    image_path = scan.get(
        "image_path"
    )

    st.subheader("🩻 Ultrasound Image")

    if image_path:

        try:

            signed_url_response = (
                supabase
                .storage
                .from_("mammosense-scans")
                .create_signed_url(
                    image_path,
                    3600,
                )
            )

            signed_url = signed_url_response.get(
                "signedURL"
            )

            if not signed_url:

                signed_url = signed_url_response.get(
                    "signedUrl"
                )

            if signed_url:

                st.image(
                    signed_url,
                    caption="Patient ultrasound",
                    use_container_width=True,
                )

            else:

                st.warning(
                    "Could not generate a secure image URL."
                )

        except Exception as error:

            st.error(
                "Could not load the ultrasound image."
            )

            st.exception(error)

    else:

        st.warning(
            "This scan does not have an image attached."
        )

        st.info(
            "Only scans uploaded after image storage "
            "was enabled will contain the original image."
        )

    # ========================================================
    # AI RESULTS
    # ========================================================

    st.divider()

    st.subheader("🤖 MammoSense Analysis")

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

    probabilities = scan.get(
        "probabilities",
        {},
    )

    if isinstance(probabilities, dict):

        st.subheader(
            "Probability Breakdown"
        )

        for name, value in probabilities.items():

            st.write(
                f"**{name}: {value:.1%}**"
            )

            st.progress(
                float(value)
            )

    # ========================================================
    # RADIOLOGIST REVIEW
    # ========================================================

    st.divider()

    st.subheader("📝 Radiologist Review")

    note = st.text_area(
        "Professional review note",
        height=180,
        placeholder=(
            "Enter your professional interpretation "
            "and recommendations..."
        ),
        key=f"review_note_{selected['id']}",
    )

    # ========================================================
    # MARK REVIEWED
    # ========================================================

    if st.button(
        "✓ Mark as Reviewed",
        type="primary",
        use_container_width=True,
        key=f"review_{selected['id']}",
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
                .update({
                    "status": "Reviewed",
                    "radiologist_id": radiologist_id,
                    "radiologist_note": note.strip(),
                })
                .eq(
                    "id",
                    selected["id"],
                )
                .execute()
            )

            st.success(
                "✅ Review completed successfully."
            )

            st.session_state.pop(
                "selected_request",
                None,
            )

            st.rerun()

        except Exception as error:

            st.error(
                "Could not save the radiologist review."
            )

            st.exception(error)
