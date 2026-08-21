import streamlit as st
from PIL import Image

from ui.background import set_background
from utils.supabase_client import get_supabase
from ai.mammosense import load_model, predict


def show_detection():

    set_background("detection.jpg")

    st.title("🧬 MammoSense")
    st.caption("AI-assisted breast ultrasound screening")

    uploaded = st.file_uploader(
        "Upload ultrasound image",
        type=["jpg", "jpeg", "png", "webp"],
        key="mammosense_upload",
    )

    if uploaded is None:
        st.info("Upload an ultrasound image to begin.")
        return

    image = Image.open(uploaded)

    st.image(
        image,
        caption="Uploaded ultrasound",
        use_container_width=True,
    )

    # ========================================================
    # ANALYSE
    # ========================================================

    if st.button(
        "Analyse with MammoSense",
        type="primary",
        use_container_width=True,
        key="analyse_mammo",
    ):

        try:

            with st.spinner("Analysing ultrasound..."):

                load_model()

                result = predict(image)

            supabase = get_supabase()

            user = supabase.auth.get_user()

            if not user.user:

                st.error("Please log in again.")

                return

            # ------------------------------------------------
            # SAVE SCAN
            # ------------------------------------------------

            response = (
                supabase
                .table("ai_scans")
                .insert({
                    "user_id": user.user.id,
                    "model": "MammoSense V2",
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                })
                .execute()
            )

            scan_id = response.data[0]["id"]

            # ------------------------------------------------
            # SAVE SESSION
            # ------------------------------------------------

            st.session_state.scan_result = result
            st.session_state.scan_id = scan_id

            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append(result)

            st.success("Analysis completed.")

        except Exception as error:

            st.error(
                "MammoSense could not analyse this image."
            )

            st.exception(error)

    # ========================================================
    # RESULT
    # ========================================================

    result = st.session_state.get("scan_result")

    if result is None:
     return
    st.divider()

    st.subheader("AI Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Finding",
            result["prediction"],
        )

    with col2:

        st.metric(
            "Confidence",
            f"{result['confidence']:.1%}",
        )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader("Probability Breakdown")

    for name, value in result["probabilities"].items():

        st.write(
            f"**{name}: {value:.1%}**"
        )

        st.progress(
            min(max(value, 0), 1)
        )

    st.divider()

    # ========================================================
    # RADIOLOGIST REVIEW
    # ========================================================

    st.subheader("👨‍⚕️ Radiologist Review")

    st.write(
        "Request a professional radiologist "
        "review of this scan."
    )

    if st.button(
        "📋 Request Radiologist Review",
        type="secondary",
        use_container_width=True,
        key="radiologist_review",
    ):

        try:

            supabase = get_supabase()

            user = supabase.auth.get_user()

            if not user.user:

                st.error("Please log in again.")

                return

            scan_id = st.session_state.get(
                "scan_id"
            )

            if not scan_id:

                st.error(
                    "No scan is available for review."
                )

                return

            # Check existing request

            existing = (
                supabase
                .table("radiologist_requests")
                .select("id,status")
                .eq(
                    "scan_id",
                    scan_id
                )
                .eq(
                    "user_id",
                    user.user.id
                )
                .execute()
                .data
            )

            if existing:

                st.info(
                    "Review already requested."
                )

                st.caption(
                    f"Status: {existing[0]['status']}"
                )

            else:

                supabase.table(
                    "radiologist_requests"
                ).insert({
                    "user_id": user.user.id,
                    "scan_id": scan_id,
                    "status": "Pending",
                }).execute()

                st.success(
                    "✅ Radiologist review requested."
                )

                st.info(
                    "Your scan is now waiting "
                    "for professional review."
                )

        except Exception as error:

            st.error(
                "Could not request radiologist review."
            )

            st.exception(error)

    st.divider()

    # ========================================================
    # CONSULTATION
    # ========================================================

    st.subheader(
        "📞 Book a Consultation"
    )

    st.write(
        "Speak with a radiologist about your scan."
    )

    call_type = st.selectbox(
        "Consultation type",
        [
            "Video call",
            "Voice call",
        ],
        key="consultation_type",
    )

    preferred_date = st.date_input(
        "Preferred date",
        key="preferred_date",
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
        ],
        key="preferred_time",
    )

    if st.button(
        "📞 Book Radiologist Consultation",
        type="primary",
        use_container_width=True,
        key="book_radiologist",
    ):

        try:

            supabase = get_supabase()

            user = supabase.auth.get_user()

            if not user.user:

                st.error("Please log in again.")

                return

            scan_id = st.session_state.get(
                "scan_id"
            )

            if not scan_id:

                st.error(
                    "No scan is associated "
                    "with this consultation."
                )

                return

            supabase.table(
                "consultations"
            ).insert({
                "user_id": user.user.id,
                "scan_id": scan_id,
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
                "Your consultation is pending "
                "radiologist confirmation."
            )

        except Exception as error:

            st.error(
                "Could not book consultation."
            )

            st.exception(error)

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "MammoSense provides AI-assisted screening "
        "information and does not replace professional "
        "medical evaluation."
    )
