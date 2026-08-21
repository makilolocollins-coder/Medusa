import streamlit as st
import pandas as pd

from ui.background import set_background
from utils.supabase_client import get_supabase


def show_health():

    set_background("health.jpg")

    supabase = get_supabase()

    # ========================================================
    # AUTHENTICATION
    # ========================================================

    response = supabase.auth.get_user()

    if not response.user:

        st.warning("Please log in.")
        return

    user_id = response.user.id

    # ========================================================
    # LOAD SCANS
    # ========================================================

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

    # ========================================================
    # LOAD RADIOLOGIST REQUESTS
    # ========================================================

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

    # ========================================================
    # PAGE HEADER
    # ========================================================

    st.title("❤️ Health Dashboard")

    st.caption(
        "Your MammoSense screening activity and "
        "professional review status"
    )

    # ========================================================
    # CALCULATE STATUS
    # ========================================================

    analyzed_count = len(scans)

    pending_reviews = [
        r for r in reviews
        if str(
            r.get("status", "")
        ).lower() == "pending"
    ]

    reviewed_reviews = [
        r for r in reviews
        if str(
            r.get("status", "")
        ).lower() == "reviewed"
    ]

    pending_count = len(
        pending_reviews
    )

    reviewed_count = len(
        reviewed_reviews
    )

    # ========================================================
    # TOP STATUS CARDS
    # ========================================================

    st.subheader("Screening Overview")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "🧬 Analyzed",
            analyzed_count
        )

        st.caption(
            "AI scans completed"
        )

    with c2:

        st.metric(
            "⏳ Pending",
            pending_count
        )

        st.caption(
            "Waiting for radiologist"
        )

    with c3:

        st.metric(
            "✅ Reviewed",
            reviewed_count
        )

        st.caption(
            "Professionally reviewed"
        )

    st.divider()

    # ========================================================
    # ANALYZED
    # ========================================================

    st.subheader("🧬 Analyzed")

    if not scans:

        st.info(
            "No scans have been analyzed yet."
        )

    else:

        df = pd.DataFrame(scans)

        total = len(df)

        average_confidence = (
            df["confidence"].mean()
        )

        latest = df.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Scans",
                total
            )

        with col2:

            st.metric(
                "Latest Finding",
                latest["prediction"]
            )

        with col3:

            st.metric(
                "Average Confidence",
                f"{average_confidence:.1%}"
            )

        st.bar_chart(
            df["prediction"].value_counts()
        )

    # ========================================================
    # PENDING REVIEWS
    # ========================================================

    st.divider()

    st.subheader(
        "⏳ Pending Radiologist Reviews"
    )

    if not pending_reviews:

        st.success(
            "No scans are currently waiting "
            "for radiologist review."
        )

    else:

        for review in pending_reviews:

            scan_id = review.get(
                "scan_id"
            )

            scan = next(
                (
                    s for s in scans
                    if s.get("id") == scan_id
                ),
                None
            )

            with st.container(
                border=True
            ):

                st.markdown(
                    "### ⏳ Review Pending"
                )

                if scan:

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            "**AI Finding**"
                        )

                        st.write(
                            scan.get(
                                "prediction",
                                "Unknown"
                            )
                        )

                    with col2:

                        st.write(
                            "**AI Confidence**"
                        )

                        st.write(
                            f"{scan.get('confidence', 0):.1%}"
                        )

                st.info(
                    "Your scan has been sent to a "
                    "radiologist and is waiting for "
                    "professional confirmation."
                )

                st.caption(
                    f"Requested: "
                    f"{review.get('created_at', 'Unknown')}"
                )

    # ========================================================
    # REVIEWED
    # ========================================================

    st.divider()

    st.subheader(
        "✅ Radiologist Reviewed"
    )

    if not reviewed_reviews:

        st.info(
            "No scans have been professionally "
            "reviewed yet."
        )

    else:

        for review in reviewed_reviews:

            scan_id = review.get(
                "scan_id"
            )

            scan = next(
                (
                    s for s in scans
                    if s.get("id") == scan_id
                ),
                None
            )

            with st.container(
                border=True
            ):

                st.markdown(
                    "### ✅ Professional Review Completed"
                )

                # --------------------------------------------
                # SCAN INFORMATION
                # --------------------------------------------

                if scan:

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            "**AI Finding**"
                        )

                        st.write(
                            scan.get(
                                "prediction",
                                "Unknown"
                            )
                        )

                    with col2:

                        st.write(
                            "**AI Confidence**"
                        )

                        st.write(
                            f"{scan.get('confidence', 0):.1%}"
                        )

                st.success(
                    "A radiologist has reviewed this scan."
                )

                # --------------------------------------------
                # RADIOLOGIST MESSAGE
                # --------------------------------------------

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
                        "the review without adding a note."
                    )

                # --------------------------------------------
                # REVIEW DATE
                # --------------------------------------------

                reviewed_at = review.get(
                    "reviewed_at"
                )

                if reviewed_at:

                    st.caption(
                        f"Reviewed on: {reviewed_at}"
                    )

    # ========================================================
    # REFRESH
    # ========================================================

    st.divider()

    if st.button(
        "🔄 Refresh Health Dashboard",
        use_container_width=True
    ):

        st.rerun()

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.caption(
        "MammoSense provides AI-assisted screening "
        "information and does not replace professional "
        "medical evaluation."
    )
