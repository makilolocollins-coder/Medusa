import streamlit as st
from pathlib import Path
import base64


def set_background(image_name):

    image_path = (
        Path(__file__).parent.parent
        / "backgrounds"
        / image_name
    )

    if not image_path.exists():
        return

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
