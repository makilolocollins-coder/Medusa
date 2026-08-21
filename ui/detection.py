import streamlit as st
from PIL import Image

from ui.background import set_background
from utils.supabase_client import get_supabase
from ai.mammosense import load_model, predict


def show_detection():

    set_background("detection.jpg")

    st.title("🧬 MammoSense")

    st.caption(
        "AI-assisted breast ultrasound screening"
    )

    uploaded = st.file_uploader(
        "Upload ultrasound image",
        type=["jpg", "jpeg", "png", "webp"],
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

    if st.button(
        "Analyse with MammoSense",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Analysing ultrasound..."
            ):

                load_model()
                result = predict(image)

            # ==================================================
            # SAVE SESSION RESULT
            # ==================================================

            st.session_state.prediction = (
                result["prediction"]
            )

            st.session_state.history.append(
                result
            )

            # ==================================================
            # SAVE TO SUPABASE
            # ==================================================

            supabase = get_supabase()

            user = supabase.auth.get_user()

            if not user.user:

                st.error(
                    "Your login session has expired."
                )

                return

            user_id = user.user.id

            response = (
                supabase
                .table("ai_scans")
                .insert({
                    "user_id": user_id,
                    "model": "MammoSense V2",
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                })
                .execute()
            )

            # Store database scan ID
            if response.data:

                st.session_state.scan_id = (
                    response.data[0]["id"]
                )

            st.success(
                "Analysis completed."
            )

        except Exception as error:

            st.error(
                "MammoSense could not analyse "
                "this image."
            )

            st.exception(error)

            return

    # ==========================================================
    # SHOW RESULT
    # ==========================================================

    if "prediction" not in st.session_state:

        return

    result = st.session_state.history[-1]

    st.divider()

    st.subheader(
        "AI Analysis"
    )

    # ==========================================================
    # MAIN RESULT
    # ==========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "AI Finding",
            result["prediction"],
        )

    with col2:

        st.metric(
            "Confidence",
            f"{result['confidence']:.1%}",
        )

    # ==========================================================
    # PROBABILITIES
    # ==========================================================

    st.subheader(
        "Probability Breakdown"
    )

    probabilities = result[
        "probabilities"
    ]

    for name, probability in probabilities.items():

        st.write(
            f"**{name} — {probability:.1%}**"
        )

        st.progress(
            min(max(probability, 0), 1)
        )

    st.caption(
        "MammoSense provides AI-assisted screening "
        "information and does not replace professional "
        "medical evaluation."
    )

    st.divider()

    # ==========================================================
    # RADIOLOGIST CONSULTATION
    # ==========================================================

    st.subheader(
        "👨‍⚕️ Radiologist Review"
    )

    st.write(
        "Would you like a qualified radiologist "
        "to review your scan?"
    )

    st.info(
        "You can request a professional review "
        "of your AI-assisted screening result."
    )

    if st.button(
        "Request Radiologist Review",
        use_container_width=True,
    ):

        try:

            scan_id = st.session_state.get(
                "scan_id"
            )

            if not scan_id:

                st.error(
                    "Scan record not found."
                )

                return

            user = (
                supabase.auth.get_user()
            )

            if not user.user:

                st.error(
                    "Please log in again."
                )

                return

            user_id = user.user.id

            # Check if request already exists

            existing = (
                supabase
                .table(
                    "radiologist_requests"
                )
                .select("*")
                .eq(
                    "scan_id",
                    scan_id
                )
                .execute()
                .data
            )

            if existing:

                st.info(
                    "Radiologist review has already "
                    "been requested for this scan."
                )

            else:

                supabase.table(
                    "radiologist_requests"
                ).insert({
                    "user_id": user_id,
                    "scan_id": scan_id,
                    "status": "Pending",
                }).execute()

                st.success(
                    "Radiologist review requested."
                )

                st.info(
                    "Your request is now pending. "
                    "A radiologist can review the scan "
                    "and provide professional confirmation."
                )

        except Exception as error:

            st.error(
                "Unable to request radiologist review."
            )

            st.exception(error)
