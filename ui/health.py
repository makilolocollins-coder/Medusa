import streamlit as st

from ui.background import set_background
from utils.supabase_client import get_supabase


def show_health():

    set_background("health.jpg")

    st.header("❤️ Health")

    supabase = get_supabase()

    user = supabase.auth.get_user()

    if not user.user:
        st.error("No logged-in user.")
        return

    scans = (
        supabase
        .table("ai_scans")
        .select("*")
        .eq("user_id", user.user.id)
        .order("created_at", desc=True)
        .execute()
        .data
    )

    st.metric(
        "Total Scans",
        len(scans)
    )

    st.divider()

    if not scans:
        st.info("No scans yet.")
        return

    for scan in scans:

        prediction = scan.get(
            "prediction",
            "Unknown"
        )

        confidence = scan.get(
            "confidence",
            0
        )

        st.write(
            f"### {prediction}"
        )

        st.write(
            f"Confidence: {confidence:.2%}"
        )

        st.caption(
            f"Model: {scan.get('model', 'Unknown')}"
        )

        st.divider()
