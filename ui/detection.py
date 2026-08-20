import streamlit as st
from PIL import Image

from ui.background import set_background
from ai.mammosense import get_mammosense


def show_detection():

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    set_background("detection.jpg")

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.header("🧬 AI Detection")

    st.write(
        "Upload a breast ultrasound image "
        "for MammoSense AI-assisted analysis."
    )

    st.divider()

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded = st.file_uploader(
        "Upload ultrasound image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded is None:

        st.info(
            "Upload a breast ultrasound image "
            "to begin analysis."
        )

        return

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded ultrasound",
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------------
    # ANALYSE BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔬 Analyse with MammoSense",
        type="primary",
        use_container_width=True,
    ):

        try:

            # ------------------------------------------------
            # LOAD MODEL
            # ------------------------------------------------

            with st.spinner(
                "Loading MammoSense AI..."
            ):

                engine = get_mammosense()

            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            with st.spinner(
                "Analysing ultrasound..."
            ):

                result = engine.predict(image)

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.success(
                "Analysis complete."
            )

            st.subheader(
                "MammoSense Result"
            )

            prediction = result[
                "prediction"
            ]

            confidence = result[
                "confidence"
            ]

            # ------------------------------------------------
            # MAIN RESULT
            # ------------------------------------------------

            st.metric(
                "AI Classification",
                prediction,
            )

            st.progress(
                confidence
            )

            st.write(
                f"Confidence: "
                f"**{confidence * 100:.2f}%**"
            )

            # ------------------------------------------------
            # CLASS PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "Class Probabilities"
            )

            probabilities = result[
                "probabilities"
            ]

            for name, probability in (
                probabilities.items()
            ):

                st.write(
                    f"**{name}**: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    probability
                )

            # ------------------------------------------------
            # MODEL INFORMATION
            # ------------------------------------------------

            with st.expander(
                "Model information"
            ):

                st.write(
                    f"**Model:** "
                    f"{result['model']}"
                )

                st.write(
                    f"**Architecture:** "
                    f"{result['architecture']}"
                )

                st.write(
                    f"**Device:** "
                    f"{result['device']}"
                )

            # ------------------------------------------------
            # MEDICAL DISCLAIMER
            # ------------------------------------------------

            st.warning(
                "AI-assisted screening only. "
                "This result is not a diagnosis and "
                "should not replace assessment by a "
                "qualified healthcare professional."
            )

        except Exception as error:

            st.error(
                "MammoSense could not analyse "
                "this image."
            )

            with st.expander(
                "Technical error"
            ):

                st.exception(error)
