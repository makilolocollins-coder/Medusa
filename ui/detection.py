import streamlit as st
from PIL import Image

from ai.mammosense import get_mammosense


def show_detection():

    st.title("AI Detection")

    st.write(
        "Upload a breast ultrasound image "
        "for MammoSense analysis."
    )

    # Load model
    try:
        with st.spinner("Loading MammoSense AI..."):
            engine = get_mammosense()

        st.success("MammoSense AI is ready.")

    except Exception as e:
        st.error("MammoSense could not be loaded.")
        st.exception(e)
        return

    # Upload image
    uploaded = st.file_uploader(
        "Upload breast ultrasound image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is None:
        return

    # Read image
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error("Could not read this image.")
        st.exception(e)
        return

    # Show image
    st.image(
        image,
        caption="Uploaded ultrasound",
        use_container_width=True,
    )

    # Analyse
    if st.button(
        "Analyse Image",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "MammoSense is analysing..."
            ):

                result = engine.predict(image)

            # Main prediction
            prediction = result["prediction"]
            confidence = result["confidence"]

            st.success(
                f"Prediction: {prediction}"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            # Probabilities
            st.subheader(
                "Class Probabilities"
            )

            for name, probability in result[
                "probabilities"
            ].items():

                st.write(
                    f"{name}: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    probability
                )

            # Model information
            with st.expander(
                "Model information"
            ):

                st.write(
                    f"Model: MammoSense V2"
                )

                st.write(
                    f"Architecture: "
                    f"{result['architecture']}"
                )

                st.write(
                    f"Device: "
                    f"{result['device']}"
                )

        except Exception as e:

            st.error(
                "MammoSense analysis failed."
            )

            st.exception(e)

    # Disclaimer
    st.warning(
        "MammoSense is an AI-assisted screening "
        "tool and is not a medical diagnosis. "
        "Results should be reviewed by a qualified "
        "healthcare professional."
    )
