import streamlit as st
from pathlib import Path
import base64


def show_home():

    # ========================================================
    # BACKGROUND
    # ========================================================

    image_path = (
        Path(__file__).parent / "background.jpg"
    )

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
                        rgba(247, 248, 250, 0.72),
                        rgba(247, 248, 250, 0.88)
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

    else:

        st.warning(
            "background.jpg was not found in the ui folder."
        )


    # ========================================================
    # HEADER
    # ========================================================

    st.title("🧬 MEDUSA AI")

    st.caption(
        "Intelligent Health Infrastructure"
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


    # ========================================================
    # AI
    # ========================================================

    st.subheader(
        "MEDUSA INTELLIGENCE"
    )

    st.info(
        "MammoSense\n\n"
        "AI-assisted breast ultrasound analysis."
    )


    # ========================================================
    # STATUS
    # ========================================================

    st.success(
        "● AI ONLINE"
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "AI-assisted screening only. "
        "Not a substitute for professional medical advice."
    )
