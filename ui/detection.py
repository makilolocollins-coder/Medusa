import uuid

import streamlit as st
from PIL import Image

from ui.background import set_background
from utils.supabase_client import get_supabase

from ai.mammosense import (
    load_model as load_mammo_model,
    predict as predict_mammo,
)

from ai.pneumonia import (
    load_model as load_pneumonia_model,
    predict as predict_pneumonia,
)


def show_detection():

    set_background("detection.jpg")

    st.title("🧬 AI Detection")
    st.caption(
        "AI-assisted medical image screening"
    )

    # ========================================================
    # MODEL SELECTOR
    # ========================================================

    model_choice = st.selectbox(
    "Select AI Model",
    [
        "MammoSense — Breast Ultrasound",
        "MammoSense Pneumonia — Chest X-ray",
    ],
    key="detection_model",
)

is_pneumonia = (
    model_choice
    == "MammoSense Pneumonia — Chest X-ray"
)

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    if is_pneumonia:

        st.info(
            "🫁 Upload a chest X-ray "
            "for pneumonia screening."
        )

        uploader_label = (
            "Upload chest X-ray"
        )

        upload_key = (
            "pneumonia_upload"
        )

        analyse_label = (
            "Analyse with Pneumonia AI"
        )

    else:

        st.info(
            "🩻 Upload a breast ultrasound "
            "for MammoSense screening."
        )

        uploader_label = (
            "Upload breast ultrasound"
        )

        upload_key = (
            "mammosense_upload"
        )

        analyse_label = (
            "Analyse with MammoSense"
        )

    # ========================================================
    # UPLOAD
    # ========================================================

    uploaded = st.file_uploader(
    "Upload chest X-ray" if is_pneumonia
    else "Upload ultrasound image",
    type=["jpg", "jpeg", "png", "webp"],
    key="medical_image_upload",
    )

    if uploaded is None:

        st.info(
            "Upload an image to begin."
        )

        return

    image = Image.open(
        uploaded
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True,
    )

    # ========================================================
    # ANALYSE
    # ========================================================

    if st.button(
        analyse_label,
        type="primary",
        use_container_width=True,
        key="analyse_selected_model",
    ):

        try:

            with st.spinner("Analysing image..."):

    if is_pneumonia:

        load_pneumonia_model()

        result = predict_pneumonia(image)

        model_name = "MammoSense Pneumonia V2"

    else:

        load_model()

        result = predict(image)

        model_name = "MammoSense V2"


            supabase = get_supabase()

            user = (
                supabase.auth.get_user()
            )

            if not user.user:

                st.error(
                    "Please log in again."
                )

                return

            user_id = user.user.id

            # =================================================
            # SAVE IMAGE
            # =================================================

            file_extension = (
                uploaded.name
                .split(".")[-1]
                .lower()
            )

            image_path = (
                f"{user_id}/"
                f"{uuid.uuid4()}."
                f"{file_extension}"
            )

            image_bytes = (
                uploaded.getvalue()
            )

            supabase.storage.from_(
                "mammosense-scans"
            ).upload(
                image_path,
                image_bytes,
                {
                    "content-type":
                        uploaded.type,

                    "upsert":
                        "false",
                },
            )

            # =================================================
            # SAVE SCAN
            # =================================================

            response = (
                supabase
                .table("ai_scans")
                .insert({
                    "user_id":
                        user_id,

                    "model":
                        model_name,

                    "prediction":
                        result["prediction"],

                    "confidence":
                        result["confidence"],

                    "probabilities":
                        result["probabilities"],

                    "image_path":
                        image_path,
                })
                .execute()
            )

            scan_id = (
                response.data[0]["id"]
            )

            # =================================================
            # SESSION
            # =================================================

            st.session_state.scan_result = (
                result
            )

            st.session_state.scan_id = (
                scan_id
            )

            st.session_state.image_path = (
                image_path
            )

            st.session_state.scan_model = (
                model_name
            )

            st.session_state.history = (
                st.session_state.get(
                    "history",
                    []
                )
            )

            st.session_state.history.append(
                result
            )

            st.success(
                "Analysis completed."
            )

        except Exception as error:

            st.error(
                "The AI model could not "
                "analyse this image."
            )

            st.exception(error)

    # ========================================================
    # RESULT
    # ========================================================

    result = st.session_state.get(
        "scan_result"
    )

    if result is None:
        return

    st.divider()

    st.subheader(
        "🤖 AI Result"
    )

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

    st.subheader(
        "Probability Breakdown"
    )

    for name, value in (
        result["probabilities"]
        .items()
    ):

        st.write(
            f"**{name}: {value:.1%}**"
        )

        st.progress(
            min(
                max(
                    float(value),
                    0,
                ),
                1,
            )
        )

    # ========================================================
    # RADIOLOGIST REVIEW
    # ========================================================

    st.divider()

    st.subheader(
        "👨‍⚕️ Radiologist Review"
    )

    st.write(
        "Request a professional "
        "radiologist review of this scan."
    )

    if st.button(
        "📋 Request Radiologist Review",
        type="secondary",
        use_container_width=True,
        key="radiologist_review",
    ):

        try:

            supabase = (
                get_supabase()
            )

            user = (
                supabase.auth.get_user()
            )

            if not user.user:

                st.error(
                    "Please log in again."
                )

                return

            scan_id = (
                st.session_state.get(
                    "scan_id"
                )
            )

            if not scan_id:

                st.error(
                    "No scan is available "
                    "for review."
                )

                return

            existing = (
                supabase
                .table(
                    "radiologist_requests"
                )
                .select(
                    "id,status"
                )
                .eq(
                    "scan_id",
                    scan_id,
                )
                .eq(
                    "user_id",
                    user.user.id,
                )
                .execute()
                .data
            )

            if existing:

                st.info(
                    "Review already requested."
                )

                st.caption(
                    f"Status: "
                    f"{existing[0]['status']}"
                )

            else:

                (
                    supabase
                    .table(
                        "radiologist_requests"
                    )
                    .insert({
                        "user_id":
                            user.user.id,

                        "scan_id":
                            scan_id,

                        "status":
                            "Pending",
                    })
                    .execute()
                )

                st.success(
                    "✅ Radiologist review requested."
                )

                st.info(
                    "Your scan is now waiting "
                    "for professional review."
                )

        except Exception as error:

            st.error(
                "Could not request "
                "radiologist review."
            )

            st.exception(error)

    # ========================================================
    # CONSULTATION
    # ========================================================

    st.divider()

    st.subheader(
        "📞 Book a Consultation"
    )

    st.write(
        "Speak with a radiologist "
        "about your scan."
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

            supabase = (
                get_supabase()
            )

            user = (
                supabase.auth.get_user()
            )

            if not user.user:

                st.error(
                    "Please log in again."
                )

                return

            scan_id = (
                st.session_state.get(
                    "scan_id"
                )
            )

            if not scan_id:

                st.error(
                    "No scan is associated "
                    "with this consultation."
                )

                return

            (
                supabase
                .table("consultations")
                .insert({
                    "user_id":
                        user.user.id,

                    "scan_id":
                        scan_id,

                    "call_type":
                        call_type,

                    "preferred_date":
                        str(
                            preferred_date
                        ),

                    "preferred_time":
                        preferred_time,

                    "status":
                        "Pending",
                })
                .execute()
            )

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
        "MammoSense provides AI-assisted "
        "screening information and does not "
        "replace professional medical evaluation."
    )
