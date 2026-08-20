import streamlit as st
from PIL import Image

from ai.lungcancer import load_model, predict


def show_detection():

    st.title("AI Detection")

    st.write(
        "Upload a medical image and select "
        "the AI model you want to test."
    )

    # ========================================================
    # MODEL SELECTOR
    # ========================================================

    model = st.selectbox(
        "Select AI Model",
        [
            "MammoSense",
            "Lung AI",
        ],
    )

    st.divider()

    # ========================================================
    # LUNG AI TEST
    # ========================================================

    if model == "Lung AI":

        st.header("🫁 Lung AI")

        st.write(
            "Upload a chest X-ray for AI-assisted "
            "lung cancer and tuberculosis analysis."
        )

        uploaded = st.file_uploader(
            "Upload chest X-ray",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            key="lung_upload",
        )

        if uploaded is None:

            st.info(
                "Upload a chest X-ray to begin."
            )

            return

        image = Image.open(
            uploaded
        )

        st.image(
            image,
            caption="Uploaded chest X-ray",
            use_container_width=True,
        )

        st.divider()

        if st.button(
            "Analyse X-ray",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Loading Lung AI..."
                ):

                    load_model()

                with st.spinner(
                    "Analysing X-ray..."
                ):

                    result = predict(
                        image
                    )

                st.success(
                    "Analysis complete."
                )

                # ------------------------------------------------
                # RESULTS
                # ------------------------------------------------

                st.subheader(
                    "AI Results"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Lung Cancer",
                        f"{result['lung_cancer'] * 100:.2f}%",
                    )

                with col2:

                    st.metric(
                        "Tuberculosis",
                        f"{result['tuberculosis'] * 100:.2f}%",
                    )

                st.subheader(
                    "Finding"
                )

                for finding in result[
                    "findings"
                ]:

                    st.info(
                        finding
                    )

                st.caption(
                    f"Model: {result['model']}"
                )

                st.caption(
                    f"Architecture: "
                    f"{result['architecture']}"
                )

            except Exception as error:

                st.error(
                    "Lung AI could not run."
                )

                st.exception(
                    error
                )

    # ========================================================
    # MAMMOSENSE
    # ========================================================

    else:

        st.header(
            "🩺 MammoSense"
        )

        st.info(
            "Your MammoSense breast ultrasound "
            "model can remain here."
        )
