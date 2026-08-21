import streamlit as st

from ui.background import set_background
from utils.supabase_client import get_supabase


# ============================================================
# HEALTH DASHBOARD
# ============================================================

def show_health():

    set_background("health.jpg")

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        """
        <div style="
            padding: 10px 0 5px 0;
        ">
            <div style="
                font-size: 14px;
                font-weight: 600;
                opacity: 0.65;
                letter-spacing: 1px;
            ">
                MEDUSA HEALTH
            </div>

            <div style="
                font-size: 34px;
                font-weight: 800;
                margin-top: 4px;
            ">
                Your Health Dashboard
            </div>

            <div style="
                font-size: 16px;
                opacity: 0.7;
                margin-top: 5px;
            ">
                Your personal AI-assisted health activity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ========================================================
    # SUPABASE
    # ========================================================

    try:

        supabase = get_supabase()

        user_response = supabase.auth.get_user()

        if not user_response.user:

            st.warning(
                "Please log in to view your health dashboard."
            )

            return

        user_id = user_response.user.id

        # ====================================================
        # LOAD SCANS
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
            "Unable to load your health information."
        )

        st.exception(error)

        return

    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    total_scans = len(scans)

    if scans:

        latest = scans[0]

        latest_prediction = latest.get(
            "prediction",
            "Unknown"
        )

        latest_confidence = latest.get(
            "confidence",
            0
        )

    else:

        latest_prediction = "None"

        latest_confidence = 0

    # ========================================================
    # SUMMARY CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Scans",
            total_scans
        )

    with col2:

        st.metric(
            "Latest Finding",
            latest_prediction
        )

    with col3:

        st.metric(
            "Confidence",
            f"{latest_confidence:.1%}"
            if scans
            else "N/A"
        )

    with col4:

        st.metric(
            "Account",
            "Active"
        )

    st.markdown("")

    # ========================================================
    # NO HISTORY
    # ========================================================

    if not scans:

        st.info(
            "You haven't completed an AI analysis yet."
        )

        st.markdown(
            "Go to **AI Detection** to perform your "
            "first MammoSense analysis."
        )

        return

    # ========================================================
    # LATEST RESULT
    # ========================================================

    st.subheader(
        "Latest AI Analysis"
    )

    prediction = latest.get(
        "prediction",
        "Unknown"
    )

    confidence = latest.get(
        "confidence",
        0
    )

    probabilities = latest.get(
        "probabilities",
        {}
    )

    with st.container(border=True):

        left, right = st.columns(
            [2, 1]
        )

        with left:

            st.markdown(
                f"### {prediction}"
            )

            st.write(
                "MammoSense V2"
            )

            if latest.get("created_at"):

                st.caption(
                    latest["created_at"]
                )

        with right:

            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )

    # ========================================================
    # PROBABILITY BREAKDOWN
    # ========================================================

    if probabilities:

        st.subheader(
            "AI Probability Breakdown"
        )

        normal = probabilities.get(
            "Normal",
            0
        )

        benign = probabilities.get(
            "Benign",
            0
        )

        malignant = probabilities.get(
            "Malignant",
            0
        )

        a, b, c = st.columns(3)

        with a:

            st.metric(
                "Normal",
                f"{normal:.1%}"
            )

            st.progress(
                min(max(normal, 0), 1)
            )

        with b:

            st.metric(
                "Benign",
                f"{benign:.1%}"
            )

            st.progress(
                min(max(benign, 0), 1)
            )

        with c:

            st.metric(
                "Malignant",
                f"{malignant:.1%}"
            )

            st.progress(
                min(max(malignant, 0), 1)
            )

    st.divider()

    # ========================================================
    # SCAN HISTORY
    # ========================================================

    st.subheader(
        "🧬 Scan History"
    )

    st.caption(
        f"{total_scans} recorded AI analyses"
    )

    for index, scan in enumerate(scans):

        prediction = scan.get(
            "prediction",
            "Unknown"
        )

        confidence = scan.get(
            "confidence",
            0
        )

        model = scan.get(
            "model",
            "Unknown"
        )

        created_at = scan.get(
            "created_at",
            ""
        )

        probabilities = scan.get(
            "probabilities",
            {}
        )

        with st.container(border=True):

            col1, col2, col3 = st.columns(
                [3, 2, 1]
            )

            with col1:

                st.markdown(
                    f"**{prediction}**"
                )

                st.caption(
                    model
                )

            with col2:

                st.write(
                    f"Confidence: "
                    f"**{confidence:.1%}**"
                )

                if created_at:

                    st.caption(
                        created_at
                    )

            with col3:

                if probabilities:

                    with st.popover(
                        "Details"
                    ):

                        st.write(
                            "Probability breakdown"
                        )

                        for name, value in probabilities.items():

                            st.write(
                                f"{name}: "
                                f"{value:.2%}"
                            )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "MammoSense provides AI-assisted screening information "
        "and is not a substitute for professional medical "
        "diagnosis or medical advice."
    )
