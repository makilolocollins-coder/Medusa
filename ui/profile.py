import streamlit as st


def show_profile():

    st.markdown(
        """
        <div class="hero">

            <h1>
                Profile
            </h1>

            <p>
                Manage your Medusa experience.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                👤 Medusa User
            </div>

            <br>

            <div class="card-text">

                Account management, privacy,
                notifications and health
                preferences will appear here.

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )
