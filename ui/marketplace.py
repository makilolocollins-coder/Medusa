import streamlit as st


def show_marketplace():

    st.markdown(
        """
        <div class="hero">

            <h1>
                Healthcare,<br>
                <span>connected.</span>
            </h1>

            <p>
                Discover healthcare professionals,
                diagnostic centres and healthcare
                services through Medusa.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            """
            <div class="market">

                <div class="market-icon">
                    👨‍⚕️
                </div>

                <div class="market-title">
                    Doctors
                </div>

                <div class="market-text">
                    Connect with qualified
                    healthcare professionals.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:

        st.markdown(
            """
            <div class="market">

                <div class="market-icon">
                    🏥
                </div>

                <div class="market-title">
                    Diagnostics
                </div>

                <div class="market-text">
                    Find imaging centres,
                    laboratories and diagnostic
                    services.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:

        st.markdown(
            """
            <div class="market">

                <div class="market-icon">
                    💊
                </div>

                <div class="market-title">
                    Pharmacy
                </div>

                <div class="market-text">
                    Access trusted healthcare
                    products and services.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "The Medusa marketplace will be connected "
        "to real providers in a future phase."
    )
