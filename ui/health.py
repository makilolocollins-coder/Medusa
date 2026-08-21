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

    scans = (
        supabase
        .table("ai_scans")
        .select("*")
        .eq("user_id", user.user.id)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )

    st.title("❤️ Health Dashboard")
    st.caption("Your personal AI health activity")

    if not scans:
        st.info("No scans yet. Start with AI Detection.")
        return

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    df = pd.DataFrame(scans)

    total = len(df)
    latest = df.iloc[0]

    average_confidence = df["confidence"].mean()

    most_common = df["prediction"].value_counts().idxmax()

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
        "Avg. Confidence",
        f"{average_confidence:.1%}"
    )

    d.metric(
        "Most Frequent",
        most_common
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

    st.divider()

    # --------------------------------------------------------
    # RECENT ACTIVITY
    # --------------------------------------------------------

    st.subheader("Recent Activity")

    history = df[
        [
            "created_at",
            "model",
            "prediction",
            "confidence",
        ]
    ].copy()

    history["confidence"] = (
        history["confidence"]
        .map(lambda x: f"{x:.1%}")
    )

    history.columns = [
        "Date",
        "Model",
        "Finding",
        "Confidence",
    ]

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "AI-assisted screening only. "
        "Not a substitute for professional medical advice."
    )
