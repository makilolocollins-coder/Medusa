import streamlit as st

from utils.supabase_client import get_supabase


def show_auth():

    supabase = get_supabase()

    st.title("🧬 MEDUSA AI")

    st.subheader("Create your account")

    email = st.text_input(
        "Email address",
        placeholder="you@example.com",
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

        if len(password) < 6:

            st.error(
                "Password must be at least 6 characters."
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
                st.session_state.auth_step = "verify"

                st.success(
                    "Account created. "
                    "Check your email to verify your account."
                )

                st.rerun()

        except Exception as error:

            st.error(
                f"Could not create account: {error}"
            )


def show_verification():

    supabase = get_supabase()

    st.title("📧 Verify your email")

    email = st.session_state.get(
        "auth_email",
        "",
    )

    st.write(
        f"We sent a verification email to **{email}**."
    )

    st.info(
        "Open the email and follow the verification "
        "instructions."
    )

    st.divider()

    st.subheader("Already verified?")

    if st.button(
        "Continue",
        type="primary",
        use_container_width=True,
    ):

        try:

            user = supabase.auth.get_user()

            if user and user.user:

                st.session_state.authenticated = True
                st.session_state.auth_step = (
                    "authenticated"
                )

                st.rerun()

            else:

                st.warning(
                    "Your email has not been verified yet."
                )

        except Exception as error:

            st.error(
                f"Could not check verification: {error}"
            )


def show_login():

    supabase = get_supabase()

    st.title("Welcome back")

    email = st.text_input(
        "Email address",
        key="login_email",
    )

    password = st.text_input(
        "Password",
        type="password",
        key="login_password",
    )

    if st.button(
        "Login",
        type="primary",
        use_container_width=True,
    ):

        try:

            response = supabase.auth.sign_in_with_password(
                {
                    "email": email,
                    "password": password,
                }
            )

            if response.user:

                st.session_state.authenticated = True
                st.session_state.auth_step = (
                    "authenticated"
                )

                st.rerun()

        except Exception as error:

            st.error(
                "Login failed. Check your email "
                "and password."
            )
