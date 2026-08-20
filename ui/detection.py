import streamlit as st
from PIL import Image

from ui.background import set_background
from ai.mammosense import load_model, predict


def show_detection():

    # ========================================================
    # BACKGROUND
    # ========================================================

    set_background("detection.jpg")

    # ========================================================
    # HEADER
    # ========================================================

    st.title("AI Detection")

    st.write(
        "Upload a breast ultrasound image "
        "and let MammoSense analyse it."
    )

    st.divider()

    # ========================================================
    # LOAD MAMMOSENSE
    # ========================================================

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

        st.exception(error)

        return

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

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

    # ========================================================
    # UPLOAD
    # ========================================================

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

        st.info(
            "Upload a breast ultrasound image "
            "to begin."
        )

        return

    # ========================================================
    # IMAGE
    # ========================================================

    image = Image.open(uploaded)

    st.image(
        image,
        caption="Uploaded ultrasound",
        use_container_width=True,
    )

    st.divider()

    # ========================================================
    # ANALYSE
    # ========================================================

    if st.button(
        "Analyse with MammoSense",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "MammoSense is analysing..."
            ):

                result = predict(image)

            st.success(
                "Analysis complete."
            )

            # =================================================
            # RESULT
            # =================================================

            st.subheader(
                "AI Result"
            )

            st.metric(
                "Prediction",
                result["prediction"],
            )

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%",
            )

            # =================================================
            # PROBABILITIES
            # =================================================

            st.subheader(
                "Class probabilities"
            )

            for name, probability in result[
                "probabilities"
            ].items():

                st.write(
                    f"**{name}** "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    probability
                )

            st.caption(
                f"Model: {result['model']}"
            )

            st.caption(
                f"Architecture: "
                f"{result['architecture']}"
            )

            st.warning(
                "AI-assisted screening only. "
                "This result is not a medical diagnosis."
            )

        except Exception as error:

            st.error(
                "MammoSense analysis failed."
            )

            st.exception(error)
