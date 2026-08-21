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
    # GET USER
    # ========================================================

    supabase = get_supabase()

    try:

        user_response = supabase.auth.get_user()

        if not user_response.user:
            st.warning("Please log in again.")
            return

        user_id = user_response.user.id

        # ====================================================
        # GET SCANS
        # ====================================================

        response = (
            supabase
            .table("ai_scans")
            .select("*")
            .eq("user_id", user_id)
            .order(
                "created_at",
                desc=True
            )
            .execute()
        )

        scans = response.data or []

    except Exception as error:

        st.error(
            "Unable to load health history."
        )

        st.exception(error)

        return

    # ========================================================
    # SUMMARY
    # ========================================================

    total_scans = len(scans)

    if scans:

        latest_result = scans[0].get(
            "prediction",
            "Unknown"
        )

    else:

        latest_result = "None"

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "AI Analyses",
            total_scans
        )

    with col2:

        st.metric(
            "Latest Result",
            latest_result
        )

    with col3:

        st.metric(
            "Account",
            "Active"
        )

    st.divider()

    # ========================================================
    # HISTORY
    # ========================================================

    st.subheader(
        "🧬 MammoSense History"
    )

    if not scans:

        st.info(
            "No AI scans have been completed yet."
        )

        return

    # ========================================================
    # DISPLAY SCANS
    # ========================================================

    for scan in scans:

        prediction = scan.get(
            "prediction",
            "Unknown"
        )

        confidence = scan.get(
            "confidence"
        )

        created_at = scan.get(
            "created_at",
            ""
        )

        if confidence is not None:

            confidence_text = (
                f"{confidence:.1%}"
            )

        else:

            confidence_text = "N/A"

        with st.container(border=True):

            left, right = st.columns(
                [3, 1]
            )

            with left:

                st.subheader(
                    prediction
                )

                st.write(
                    "MammoSense V2"
                )

                st.caption(
                    created_at
                )

            with right:

                st.metric(
                    "Confidence",
                    confidence_text
                )
