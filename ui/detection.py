import streamlit as st

from ui.background import set_background


def show_detection():

    set_background("detection.jpg")

    st.header("🧬 AI Detection")

    st.write(
        "Upload a breast ultrasound image "
        "for MammoSense analysis."
    )

    st.divider()

    uploaded = st.file_uploader(
        "Upload ultrasound image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded is not None:

        st.image(
            uploaded,
            caption="Uploaded ultrasound",
            use_container_width=True,
        )

        st.info(
            "MammoSense analysis will appear here."
        )
