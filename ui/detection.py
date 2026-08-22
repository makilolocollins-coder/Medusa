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

        uploader_label = "Upload chest X-ray"

        analyse_label = (
            "Analyse with Pneumonia AI"
        )

        model_name = (
            "MammoSense Pneumonia V2"
        )

    else:

        st.info(
            "🩻 Upload a breast ultrasound "
            "for MammoSense screening."
        )

        uploader_label = (
            "Upload breast ultrasound"
        )

        analyse_label = (
            "Analyse with MammoSense"
        )

        model_name = "MammoSense V2"

    # ========================================================
    # UPLOAD
    # ========================================================

    uploaded = st.file_uploader(
        uploader_label,
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="medical_image_upload",
    )

    if uploaded is None:

        st.info(
            "Upload an image to begin."
        )

        return

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    try:

        image = Image.open(
            uploaded
        ).convert("RGB")

    except Exception as error:

        st.error(
            "The uploaded file could not "
            "be opened as an image."
        )

        st.exception(error)

        return

    st.image(
        image,
        caption=(
            "Uploaded chest X-ray"
            if is_pneumonia
            else "Uploaded breast ultrasound"
        ),
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

            with st.spinner(
                "Analysing image..."
            ):

                # --------------------------------------------
                # PNEUMONIA MODEL
                # --------------------------------------------

                if is_pneumonia:

                    load_pneumonia_model()

                    result = predict_pneumonia(
                        image
                    )

                # --------------------------------------------
                # BREAST ULTRASOUND MODEL
                # --------------------------------------------

                else:

                    load_mammo_model()

                    result = predict_mammo(
                        image
                    )

            # =================================================
            # SUPABASE
            # =================================================

            supabase = get_supabase()

            user_response = (
                supabase.auth.get_user()
            )

            if not user_response.user:

                st.error(
                    "Please log in again."
                )

                return

            user_id = (
                user_response.user.id
            )

            # =================================================
            # SAVE IMAGE TO STORAGE
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
                    "content-type": (
                        uploaded.type
                    ),
                    "upsert": "false",
                },
            )

            # =================================================
            # SAVE SCAN
            # =================================================

            response = (
                supabase
                .table("ai_scans")
                .insert({
                    "user_id": user_id,

                    "model": model_name,

                    "prediction": (
                        result["prediction"]
                    ),

                    "confidence": (
                        result["confidence"]
                    ),

                    "probabilities": (
                        result["probabilities"]
                    ),

                    "image_path": image_path,
                })
                .execute()
            )

            if not response.data:

                st.error(
                    "The scan was analysed but "
                    "could not be saved."
                )

                return

            scan_id = (
                response.data[0]["id"]
            )

            # =================================================
            # SESSION STATE
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

            st.session_state.scan_type = (
                "pneumonia"
                if is_pneumonia
                else "mammosense"
            )

            if "history" not in (
                st.session_state
            ):

                st.session_state.history = []

            st.session_state.history.append({
                "model": model_name,
                "result": result,
                "scan_id": scan_id,
            })

            st.success(
                "✅ Analysis completed."
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

    current_model = (
        st.session_state.get(
            "scan_model",
            model_name,
        )
    )

    st.caption(
        f"Model: {current_model}"
    )

    # ========================================================
    # MAIN RESULT
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Finding",
            result.get(
                "prediction",
                "Unknown",
            ),
        )

    with col2:

        confidence = float(
            result.get(
                "confidence",
                0,
            )
        )

        st.metric(
            "Confidence",
            f"{confidence:.1%}",
        )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    probabilities = result.get(
        "probabilities",
        {},
    )

    if isinstance(
        probabilities,
        dict,
    ) and probabilities:

        st.subheader(
            "Probability Breakdown"
        )

        for name, value in (
            probabilities.items()
        ):

            value = float(value)

            st.write(
                f"**{name}: {value:.1%}**"
            )

            st.progress(
                min(
                    max(
                        value,
                        0.0,
                    ),
                    1.0,
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

            supabase = get_supabase()

            user_response = (
                supabase.auth.get_user()
            )

            if not user_response.user:

                st.error(
                    "Please log in again."
                )

                return

            current_user_id = (
                user_response.user.id
            )

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
                    current_user_id,
                )
                .execute()
                .data
                or []
            )

            if existing:

                status = existing[0].get(
                    "status",
                    "Unknown",
                )

                st.info(
                    "Review already requested."
                )

                st.caption(
                    f"Status: {status}"
                )

            else:

                (
                    supabase
                    .table(
                        "radiologist_requests"
                    )
                    .insert({
                        "user_id":
                            current_user_id,

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

            supabase = get_supabase()

            user_response = (
                supabase.auth.get_user()
            )

            if not user_response.user:

                st.error(
                    "Please log in again."
                )

                return

            current_user_id = (
                user_response.user.id
            )

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
                        current_user_id,

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
