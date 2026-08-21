import streamlit as st
import pandas as pd

from ui.background import set_background
from utils.supabase_client import get_supabase


def show_health():

    set_background("health.jpg")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    supabase = get_supabase()
    user = supabase.auth.get_user()

    if not user.user:
        st.warning("Please log in.")
        return

    user_id = user.user.id

    # --------------------------------------------------------
    # LOAD SCANS
    # --------------------------------------------------------

    scans = (
        supabase
        .table("ai_scans")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )

    # --------------------------------------------------------
    # LOAD RADIOLOGIST REVIEWS
    # --------------------------------------------------------

    try:

        reviews = (
            supabase
            .table("radiologist_requests")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
            .data
            or []
        )

    except Exception as error:

        st.error(
            "Could not load radiologist reviews."
        )

        st.exception(error)

        reviews = []

    # --------------------------------------------------------
    # REVIEW COUNTS
    # --------------------------------------------------------

    total_reviews = len(reviews)

    pending_reviews = sum(
        1
        for review in reviews
        if str(
            review.get("status", "")
        ).lower() == "pending"
    )

    reviewed_cases = sum(
        1
        for review in reviews
        if str(
            review.get("status", "")
        ).lower() == "reviewed"
    )

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.title("❤️ Health Dashboard")
    st.caption(
        "Your personal AI health activity"
    )

    # --------------------------------------------------------
    # NO SCANS
    # --------------------------------------------------------

    if not scans:

        st.info(
            "No scans yet. Start with AI Detection."
        )

        # Still show review statistics
        # if review records exist.

        if total_reviews > 0:

            st.divider()

            st.subheader(
                "👨‍⚕️ Radiologist Reviews"
            )

            a, b, c = st.columns(3)

            a.metric(
                "Total Reviews",
                total_reviews
            )

            b.metric(
                "Pending",
                pending_reviews
            )

            c.metric(
                "Reviewed",
                reviewed_cases
            )

        return

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    df = pd.DataFrame(scans)

    total = len(df)

    latest = df.iloc[0]

    average_confidence = df[
        "confidence"
    ].mean()

    most_common = (
        df["prediction"]
        .value_counts()
        .idxmax()
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    a, b, c, d = st.columns(4)

    a.metric(
        "Total Scans",
        total
    )

    b.metric(
        "Latest Result",
        latest["prediction"]
    )

    c.metric(
        "Pending Reviews",
        pending_reviews
    )

    d.metric(
        "Reviewed Cases",
        reviewed_cases
    )

    st.divider()

    # --------------------------------------------------------
    # CHART + LATEST RESULT
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("Findings")

        counts = (
            df["prediction"]
            .value_counts()
        )

        st.bar_chart(counts)

    with right:

        st.subheader("Latest Analysis")

        st.metric(
            "Finding",
            latest["prediction"]
        )

        st.metric(
            "Confidence",
            f"{latest['confidence']:.1%}"
        )

        st.caption(
            f"Model: {latest['model']}"
        )

    # ========================================================
    # RADIOLOGIST REVIEWS
    # ========================================================

    st.divider()

    st.subheader(
        "👨‍⚕️ Radiologist Reviews"
    )

    st.caption(
        "Professional reviews and messages "
        "for your submitted scans."
    )

    # --------------------------------------------------------
    # REVIEW SUMMARY
    # --------------------------------------------------------

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Reviews Requested",
        total_reviews
    )

    r2.metric(
        "Pending",
        pending_reviews
    )

    r3.metric(
        "Completed",
        reviewed_cases
    )

    # --------------------------------------------------------
    # REVIEW CARDS
    # --------------------------------------------------------

    if not reviews:

        st.info(
            "You have not requested a radiologist "
            "review yet."
        )

    else:

        for review in reviews:

            status = str(
                review.get("status", "")
            ).lower()

            scan_id = review.get(
                "scan_id"
            )

            # -----------------------------------------------
            # FIND ASSOCIATED SCAN
            # -----------------------------------------------

            associated_scan = None

            for scan in scans:

                if scan.get("id") == scan_id:

                    associated_scan = scan

                    break

            # -----------------------------------------------
            # REVIEW CARD
            # -----------------------------------------------

            with st.container(
                border=True
            ):

                if status == "reviewed":

                    st.markdown(
                        "### ✅ Radiologist Review Completed"
                    )

                elif status == "pending":

                    st.markdown(
                        "### ⏳ Radiologist Review Pending"
                    )

                else:

                    st.markdown(
                        "### 📋 Radiologist Review"
                    )

                # -------------------------------------------
                # SCAN INFORMATION
                # -------------------------------------------

                if associated_scan:

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            "**AI Finding**"
                        )

                        st.write(
                            associated_scan.get(
                                "prediction",
                                "Unknown"
                            )
                        )

                    with col2:

                        st.write(
                            "**AI Confidence**"
                        )

                        st.write(
                            f"{associated_scan.get('confidence', 0):.1%}"
                        )

                # -------------------------------------------
                # PENDING
                # -------------------------------------------

                if status == "pending":

                    st.info(
                        "Your scan is waiting for "
                        "radiologist confirmation."
                    )

                    st.caption(
                        "The professional review will "
                        "appear here once completed."
                    )

                # -------------------------------------------
                # REVIEWED
                # -------------------------------------------

                elif status == "reviewed":

                    st.success(
                        "Your scan has been reviewed "
                        "by a radiologist."
                    )

                    # ---------------------------------------
                    # RADIOLOGIST MESSAGE
                    # ---------------------------------------

                    note = review.get(
                        "radiologist_note"
                    )

                    if note:

                        st.markdown(
                            "#### 📝 Radiologist Message"
                        )

                        st.info(
                            note
                        )

                    else:

                        st.caption(
                            "The radiologist completed "
                            "the review without adding "
                            "a message."
                        )

                    # ---------------------------------------
                    # REVIEW DATE
                    # ---------------------------------------

                    reviewed_at = review.get(
                        "reviewed_at"
                    )

                    if reviewed_at:

                        st.caption(
                            f"Reviewed: {reviewed_at}"
                        )

                else:

                    st.warning(
                        f"Review status: "
                        f"{review.get('status', 'Unknown')}"
                    )

    # ========================================================
    # MEDICAL DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "MammoSense provides AI-assisted screening "
        "information. AI results do not replace "
        "professional medical evaluation."
    )
