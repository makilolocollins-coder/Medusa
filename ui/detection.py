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

            supabase = get_supabase()

            user = supabase.auth.get_user()

            if not user.user:

                st.error(
                    "Please log in again."
                )

                return

            user_id = user.user.id

            # ----------------------------------------------
            # SAVE SCAN
            # ----------------------------------------------

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

            scan_id = response.data[0]["id"]

            st.session_state.prediction = (
                result["prediction"]
            )

            st.session_state.history.append(
                result
            )

            st.session_state.scan_id = scan_id

            st.session_state.scan_result = result

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

    # ========================================================
    # RESULT
    # ========================================================

    if "scan_result" not in st.session_state:
        return

    result = st.session_state.scan_result

    st.divider()

    st.subheader(
        "AI Result"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Finding",
            result["prediction"]
        )

    with col2:

        st.metric(
            "Confidence",
            f"{result['confidence']:.1%}"
        )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader(
        "Probability Breakdown"
    )

    for name, value in result[
        "probabilities"
    ].items():

        st.write(
            f"**{name}: {value:.1%}**"
        )

        st.progress(
            min(max(value, 0), 1)
        )

    st.divider()

    # ========================================================
    # CONSULTATION
    # ========================================================

    st.subheader(
        "👨‍⚕️ Radiologist Consultation"
    )

    st.write(
        "Would you like a radiologist to review "
        "your ultrasound result?"
    )

    call_type = st.selectbox(
        "Consultation type",
        [
            "Video call",
            "Voice call",
        ]
    )

    preferred_date = st.date_input(
        "Preferred date"
    )

    preferred_time = st.selectbox(
        "Preferred time",
        [
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "14:00",
            "15:00",
            "16:00",
        ]
    )

    if st.button(
        "📞 Book Radiologist Consultation",
        type="primary",
        use_container_width=True,
    ):

        try:

            supabase = get_supabase()

            user = supabase.auth.get_user()

            if not user.user:

                st.error(
                    "Please log in again."
                )

                return

            user_id = user.user.id

            supabase.table(
                "consultations"
            ).insert({
                "user_id": user_id,
                "scan_id": st.session_state.scan_id,
                "call_type": call_type,
                "preferred_date": str(
                    preferred_date
                ),
                "preferred_time": preferred_time,
                "status": "Pending",
            }).execute()

            st.success(
                "✅ Consultation request submitted."
            )

            st.info(
                "Your request is pending radiologist "
                "review. You will be notified when "
                "your consultation is confirmed."
            )

        except Exception as error:

            st.error(
                "Unable to book consultation."
            )

            st.exception(error)

    st.divider()

    st.caption(
        "MammoSense provides AI-assisted screening "
        "information and does not replace professional "
        "medical evaluation."
    )
