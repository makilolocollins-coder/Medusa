import streamlit as st

from ui.background import set_background
from utils.supabase_client import get_supabase


def show_health():

    set_background("health.jpg")

    st.header("❤️ Health")

    st.write("Testing Supabase health history...")

    try:

        supabase = get_supabase()

        st.success("Supabase client loaded.")

        user_response = supabase.auth.get_user()

        st.write(
            "Authenticated user:",
            user_response.user.id
            if user_response.user
            else "NO USER",
        )

        if not user_response.user:
            st.error("No logged-in Supabase user.")
            return

        user_id = user_response.user.id

        response = (
            supabase
            .table("ai_scans")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        st.success("Successfully connected to ai_scans.")

        st.write("Number of scans:", len(response.data))

        st.write("Database response:")

        st.json(response.data)

    except Exception as error:

        st.error("Health history failed.")

        st.exception(error)
