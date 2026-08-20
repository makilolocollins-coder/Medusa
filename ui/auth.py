import streamlit as st

from utils.supabase_client import get_supabase


def show_auth():

    supabase = get_supabase()

    st.title("🧬 MEDUSA AI")

    st.subheader("Create your account")

    email = st.text_input(
        "Email address"
    )

    password = st.text_input(
        "Password",
        type="password",
    )

    if st.button(
        "Create account",
        type="primary",
        use_container_width=True,
    ):

        if not email or not password:

            st.error(
                "Enter your email and password."
            )

            return

        try:

            response = supabase.auth.sign_up(
                {
                    "email": email,
                    "password": password,
                }
            )

            if response.user:

                st.session_state.auth_email = email
                st.session_state.verify_email = True

                st.success(
                    "Verification code sent to your email."
                )

                st.rerun()

        except Exception as error:

            st.error(
                f"Could not create account: {error}"
            )


def show_verification():

    supabase = get_supabase()

    st.title("Verify your email")

    email = st.session_state.get(
        "auth_email",
        "",
    )

    st.write(
        f"We sent a verification code to **{email}**."
    )

    code = st.text_input(
        "Enter verification code",
        max_chars=6,
    )

    if st.button(
        "Verify email",
        type="primary",
        use_container_width=True,
    ):

        if len(code) != 6:

            st.error(
                "Enter the 6-digit code."
            )

            return

        try:

            response = supabase.auth.verify_otp(
                {
                    "email": email,
                    "token": code,
                    "type": "email",
                }
            )

            if response.user:

                st.session_state.authenticated = True
                st.session_state.verify_email = False

                st.success(
                    "Email verified successfully."
                )

                st.rerun()

        except Exception as error:

            st.error(
                f"Invalid verification code: {error}"
            )
