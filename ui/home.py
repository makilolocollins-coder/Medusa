import streamlit as st
from pathlib import Path
import base64


def show_home():

    # ========================================================
    # BACKGROUND
    # ========================================================

    image_path = Path(__file__).parent.parent / "background.jpg"

    if image_path.exists():

        with open(image_path, "rb") as file:

            encoded = base64.b64encode(
                file.read()
            ).decode()

        st.markdown(
            f"""
            <style>

            .stApp {{
                background-image:
                    linear-gradient(
                        rgba(247,248,250,0.72),
                        rgba(247,248,250,0.90)
                    ),
                    url(
                        "data:image/jpeg;base64,{encoded}"
                    );

                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            </style>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # HERO
    # ========================================================

    st.header(
        "Your health, intelligently connected."
    )

    st.write(
        "Medusa combines artificial intelligence, "
        "health insights and healthcare services "
        "in one intelligent platform."
    )

    st.write("")


    # ========================================================
    # AI DETECTION
    # ========================================================

    st.subheader(
        "🧠 Medusa Intelligence"
    )

    st.info(
        "MammoSense\n\n"
        "AI-assisted breast ultrasound analysis."
    )


    if st.button(
        "Start AI Analysis →",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.page = "AI Detection"

        st.rerun()


    # ========================================================
    # MODELS
    # ========================================================

    st.subheader("AI Models")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🧬 MammoSense")

        st.write(
            "Breast ultrasound AI "
            "classification."
        )

        st.success("Available")


    with col2:

        st.markdown("### 🧠 Prostate AI")

        st.write(
            "Multimodal prostate MRI "
            "intelligence."
        )

        st.info("Coming soon")


    with col3:

        st.markdown("### ✦ More AI")

        st.write(
            "Additional medical AI "
            "systems."
        )

        st.info("Coming soon")


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "AI-assisted screening only. "
        "Not a substitute for professional medical advice."
    )
