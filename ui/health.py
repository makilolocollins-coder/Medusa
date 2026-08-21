import streamlit as st

from ui.background import set_background
from utils.supabase_client import get_supabase


def show_health():

    set_background("health.jpg")

    st.header("❤️ Health")

    st.write(
        "Your health information and AI analysis history."
    )

    st.divider()

    # ========================================================
    # SUPABASE
    # ========================================================

    supabase = get_supabase()

    try:

        user_response = supabase.auth.get_user()

        if not user_response.user:

            st.warning(
                "Please log in to view your health history."
            )

            return

        user_id = user_response.user.id

        # ====================================================
        # LOAD USER SCANS
        # ====================================================

        response = (
            supabase
            .table("ai_scans")
            .select("*")
            .eq("user_id", user_id)
            .order(
                "created_at",
                desc=True,
            )
            .execute()
        )

        scans = response.data or []

    except Exception as error:

        st.error(
            "Unable to load your health history."
        )

        st.exception(error)

        return

    # ========================================================
    # SUMMARY
    # ========================================================

    latest = (
        scans[0]["prediction"]
        if scans
        else "None"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "AI Analyses",
            len(scans),
        )

    with col2:

        st.metric(
            "Latest Result",
            latest,
        )

    with col3:

        st.metric(
            "Status",
            "Active",
        )

    st.divider()

    # ========================================================
    # HISTORY
    # ========================================================

    st.subheader(
        "Analysis History"
    )

    if not scans:

        st.info(
            "No AI analysis has been completed yet."
        )

        return

    # ========================================================
    # DISPLAY SCANS
    # ========================================================

    for scan in scans:

        prediction = scan.get(
            "prediction",
            "Unknown",
        )

        confidence = scan.get(
            "confidence"
        )

        model = scan.get(
            "model",
            "Unknown model",
        )

        created_at = scan.get(
            "created_at",
            "",
        )

        if confidence is not None:

            confidence_text = (
                f"{confidence:.1%}"
            )

        else:

            confidence_text = "N/A"

        with st.container(
            border=True
        ):

            st.write(
                f"### {prediction}"
            )

            st.write(
                f"**Model:** {model}"
            )

            st.write(
                f"**Confidence:** "
                f"{confidence_text}"
            )

            if created_at:

                st.caption(
                    created_at
                )
