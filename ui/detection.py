import streamlit as st
import textwrap
from PIL import Image

from ai.mammosense import (
    load_model,
    predict,
)


def show_detection():

    st.success("MEDUSA BUILD: V2-2026-08-19")

    st.markdown(
        textwrap.dedent("""
        <div class="hero">

            <h1>
                AI Detection
            </h1>

            <p>
                Upload a breast ultrasound image
                and let MammoSense analyse it.
            </p>

        </div>
        """),
        unsafe_allow_html=True,
    )

    try:

        with st.spinner(
            "Loading MammoSense AI..."
        ):

            package = load_model()

        st.success(
            "MammoSense AI is ready."
        )

    except Exception as error:

        st.error(
            "MammoSense could not be loaded."
        )

        st.code(
            str(error)
        )

        return

    with st.expander(
        "Model information"
    ):

        st.write(
            f"**Model:** "
            f"{package['model_file']}"
        )

        st.write(
            f"**Architecture:** "
            f"{package['architecture']}"
        )

        st.write(
            f"**Classes:** "
            f"{', '.join(package['classes'])}"
        )

        st.write(
            f"**Device:** "
            f"{package['device']}"
        )

    uploaded = st.file_uploader(
        "Upload breast ultrasound image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded is None:
        return

    image = Image.open(
        uploaded
    )

    left, right = st.columns(
        [1.2, 1]
    )

    with left:

        st.image(
            image,
            caption="Ultrasound image",
            use_container_width=True,
        )
