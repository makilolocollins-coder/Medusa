import streamlit as st
from PIL import Image

from ui.background import set_background
from ai.mammosense import get_mammosense
from utils.supabase_client import get_supabase

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
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is None:
        st.info(
            "Upload a breast ultrasound image "
            "to begin analysis."
        )
        return

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded ultrasound",
        use_container_width=True,
    )

    st.divider()

    if st.button(
        "🔬 Analyse with MammoSense",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner("Loading MammoSense AI..."):
                engine = get_mammosense()

            with st.spinner("Analysing ultrasound..."):
                result = engine.predict(image)

            st.success("Analysis complete.")

            st.subheader("MammoSense Result")

            prediction = result["prediction"]
            confidence = result["confidence"]

            st.metric(
                "AI Classification",
                prediction,
            )

            st.write(
                f"Confidence: **{confidence * 100:.2f}%**"
            )

            st.progress(confidence)

            st.subheader("Class Probabilities")

            for name, probability in result[
                "probabilities"
            ].items():

                st.write(
                    f"**{name}:** "
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)

            with st.expander("Model information"):

                st.write(
                    f"**Model:** {result['model']}"
                )

                st.write(
                    f"**Architecture:** "
                    f"{result['architecture']}"
                )

                st.write(
                    f"**Device:** {result['device']}"
                )

            st.warning(
                "AI-assisted screening only. "
                "This is not a medical diagnosis."
            )

        except Exception as error:

            st.error(
                "MammoSense could not analyse this image."
            )

            st.exception(error)
