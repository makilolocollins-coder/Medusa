import streamlit as st


def show_health():

    st.markdown(
        """
        <div class="hero">

            <h1>
                Your Health
            </h1>

            <p>
                View your Medusa AI analysis history.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.history

    if not history:

        st.info(
            "No AI analysis has been performed yet."
        )

        return

    for item in reversed(history):

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    🧬 MammoSense Analysis
                </div>

                <div class="card-text">

                    Result:
                    <b>{item["prediction"]}</b>

                    <br>

                    Confidence:
                    {item["confidence"] * 100:.2f}%

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )
