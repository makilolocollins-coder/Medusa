import streamlit as st


def show_profile():

    st.header("👤 Profile")

    st.write(
        "Manage your Medusa profile and "
        "application preferences."
    )


    # ========================================================
    # PROFILE
    # ========================================================

    name = st.text_input(
        "Name",
        placeholder="Enter your name",
    )

    email = st.text_input(
        "Email",
        placeholder="Enter your email",
    )


    # ========================================================
    # PREFERENCES
    # ========================================================

    st.subheader("Preferences")

    notifications = st.toggle(
        "Enable notifications",
        value=True,
    )

    save = st.button(
        "Save Profile",
        type="primary",
    )


    if save:

        st.success(
            "Profile preferences saved."
        )


    # ========================================================
    # ACCOUNT
    # ========================================================

    st.divider()

    st.subheader("Account")

    st.write(
        f"Notifications: "
        f"{'Enabled' if notifications else 'Disabled'}"
    )

    st.caption(
        "Medusa keeps your health experience "
        "organized in one place."
    )
