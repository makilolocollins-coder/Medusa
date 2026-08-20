import streamlit as st

from ui.background import set_background


def show_profile():

    set_background("profile.jpg")

    st.header("👤 Profile")

    st.write(
        "Manage your Medusa profile and "
        "application preferences."
    )

    st.divider()

    st.subheader("Personal Information")

    name = st.text_input(
        "Name",
        placeholder="Enter your name",
    )

    email = st.text_input(
        "Email",
        placeholder="Enter your email",
    )

    st.divider()

    st.subheader("Preferences")

    notifications = st.toggle(
        "Enable notifications",
        value=True,
    )

    if st.button(
        "Save Profile",
        type="primary",
    ):

        st.success(
            "Profile preferences saved."
        )

    st.divider()

    st.subheader("Account")

    st.write(
        "Notifications: "
        + (
            "Enabled"
            if notifications
            else "Disabled"
        )
    )

    st.caption(
        "Medusa keeps your health experience "
        "organized in one place."
    )
