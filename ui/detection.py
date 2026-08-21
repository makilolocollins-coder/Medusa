import streamlit as st
from PIL import Image

from ui.background import set_background
from ai.mammosense import load_model, predict
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
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded is None:

        st.info(
            "Upload an ultrasound image to begin."
        )

        return

    image = Image.open(uploaded)

    st.image(
        image,
        caption="Uploaded ultrasound",
        use_container_width=True,
    )

    st.divider()

    if st.button(
        "Analyse with MammoSense",
        type="primary",
        use_container_width=True,
    ):

        try:

            # ==================================================
            # RUN MODEL
            # ==================================================

            with st.spinner(
                "Loading MammoSense..."
            ):

                load_model()

            with st.spinner(
                "Analysing ultrasound..."
            ):

                result = predict(image)

            # ==================================================
            # RESULT
            # ==================================================

            st.success(
                "Analysis complete."
            )

            st.subheader(
                "AI Result"
            )

            st.metric(
                "Finding",
                result["prediction"],
            )

            st.metric(
                "Confidence",
                f"{result['confidence']:.2%}",
            )

            # ==================================================
            # PROBABILITIES
            # ==================================================

            st.subheader(
                "Class Probabilities"
            )

            for name, probability in result[
                "probabilities"
            ].items():

                st.write(
                    f"**{name}**: "
                    f"{probability:.2%}"
                )

            # ==================================================
            # SESSION HISTORY
            # ==================================================

            st.session_state.prediction = (
                result["prediction"]
            )

            st.session_state.history.append(
                result
            )

            # ==================================================
            # SUPABASE
            # ==================================================

            supabase = get_supabase()

            user_response = (
                supabase.auth.get_user()
            )

            if not user_response.user:

                st.warning(
                    "No authenticated user found. "
                    "The scan was not saved."
                )

                return

            user_id = user_response.user.id

            # ==================================================
            # SAVE SCAN
            # ==================================================

            supabase.table(
                "ai_scans"
            ).insert(
                {
                    "user_id": user_id,

                    "model": "MammoSense V2",

                    "prediction":
                        result["prediction"],

                    "confidence":
                        result["confidence"],

                    "probabilities":
                        result["probabilities"],
                }
            ).execute()

            st.success(
                "✅ Scan saved to your Health history."
            )

        except Exception as error:

            st.error(
                "MammoSense analysis failed."
            )

            st.exception(error)
