import io
import uuid
from datetime import datetime

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


STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi",
    "Bayelsa", "Benue", "Borno", "Cross River", "Delta",
    "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo",
    "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi",
    "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo",
    "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba",
    "Yobe", "Zamfara", "FCT"
]


def user():
    try:
        return get_supabase().auth.get_user().user
    except Exception:
        return None


def init():
    defaults = {
        "patient_id": None,
        "patient_name": "",
        "scan_id": None,
        "scan_result": None,
        "scan_image": None,
        "review_status": "NOT_REQUESTED",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def new_patient_id():
    return (
        f"MED-P-{datetime.now():%Y%m%d}-"
        f"{uuid.uuid4().hex[:8].upper()}"
    )


def reset():
    for k in [
        "patient_id",
        "scan_id",
        "scan_result",
        "scan_image",
    ]:
        st.session_state[k] = None

    st.session_state.review_status = "NOT_REQUESTED"


def review_status(scan_id):
    if not scan_id:
        return "NOT_REQUESTED"

    sb = get_supabase()

    try:
        req = (
            sb.table("radiologist_requests")
            .select("status")
            .eq("scan_id", scan_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )

        if req:
            status = str(req[0]["status"]).upper()

            if status in ["PENDING", "REQUESTED"]:
                return "PENDING"

            if status in ["APPROVED", "COMPLETED", "REVIEWED"]:
                return "APPROVED"

        rev = (
            sb.table("radiologist_reviews")
            .select("status,approved")
            .eq("scan_id", scan_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )

        if rev:
            if (
                str(rev[0].get("status", "")).upper()
                == "APPROVED"
                or rev[0].get("approved") is True
            ):
                return "APPROVED"

    except Exception:
        pass

    return "NOT_REQUESTED"


def show_detection():

    init()
    set_background("detection.jpg")

    st.title("Medusa AI")
    st.caption("AI-assisted medical imaging and radiologist review")

    # ==========================================================
    # PATIENT REGISTRATION
    # ==========================================================

    st.subheader("Patient Registration")

    name = st.text_input(
        "Patient full name",
        value=st.session_state.patient_name,
        placeholder="Enter patient's full name",
    ).strip()

    state = st.selectbox(
        "Patient state",
        STATES,
        index=STATES.index("Delta"),
    )

    if name != st.session_state.patient_name:
        st.session_state.patient_name = name

    if name and not st.session_state.patient_id:
        st.session_state.patient_id = new_patient_id()

    if st.session_state.patient_id:
        st.info(f"Patient ID: {st.session_state.patient_id}")

    if not name:
        st.warning("Enter the patient's full name to continue.")
        return

    # ==========================================================
    # NEW EXAMINATION
    # ==========================================================

    if st.session_state.scan_result:

        if st.button(
            "Start New Examination",
            use_container_width=True,
        ):
            reset()
            st.rerun()

    # ==========================================================
    # EXAMINATION TYPE
    # ==========================================================

    st.subheader("Medical Examination")

    model_choice = st.selectbox(
        "Select examination",
        [
            "Breast Ultrasound",
            "Chest X-ray",
        ],
    )

    pneumonia = model_choice == "Chest X-ray"

    model_name = (
        "MammoSense Pneumonia V2"
        if pneumonia
        else "MammoSense V2"
    )

    # ==========================================================
    # UPLOAD
    # ==========================================================

    uploaded = st.file_uploader(
        "Upload medical scan",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        image_bytes = uploaded.getvalue()
        st.session_state.scan_image = image_bytes
    else:
        image_bytes = st.session_state.scan_image

    if not image_bytes:
        st.info("Upload a medical scan to continue.")
        return

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        st.image(
            image,
            caption=model_choice,
            use_container_width=True,
        )

    except Exception:
        st.error("Invalid medical image.")
        return

    # ==========================================================
    # AI ANALYSIS
    # ==========================================================

    if st.button(
        "Analyze Examination",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner("Medusa AI is analyzing the scan..."):

                if pneumonia:
                    load_pneumonia_model()
                    result = predict_pneumonia(image)
                else:
                    load_mammo_model()
                    result = predict_mammo(image)

            current_user = user()

            if not current_user:
                st.error("Login session expired.")
                return

            sb = get_supabase()

            ext = (
                uploaded.name.split(".")[-1].lower()
                if uploaded
                else "png"
            )

            path = (
                f"{current_user.id}/"
                f"{st.session_state.patient_id}/"
                f"{uuid.uuid4().hex}.{ext}"
            )

            sb.storage.from_("mammosense-scans").upload(
                path,
                image_bytes,
                {
                    "content-type": uploaded.type
                    if uploaded
                    else "image/png",
                    "upsert": "false",
                },
            )

            saved = (
                sb.table("ai_scans")
                .insert({
                    "user_id": current_user.id,
                    "patient_id": st.session_state.patient_id,
                    "patient_name": name,
                    "patient_state": state,
                    "examination": model_choice,
                    "model": model_name,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": result.get(
                        "probabilities",
                        {},
                    ),
                    "image_path": path,
                    "status": "AI_COMPLETED",
                })
                .execute()
            )

            if not saved.data:
                st.error("Could not save examination.")
                return

            st.session_state.scan_id = saved.data[0]["id"]
            st.session_state.scan_result = result
            st.session_state.review_status = "NOT_REQUESTED"

            st.success("AI analysis completed.")
            st.rerun()

        except Exception as e:
            st.error("AI analysis failed.")
            st.exception(e)
            return

    # ==========================================================
    # RESULT
    # ==========================================================

    result = st.session_state.scan_result

    if not result:
        return

    status = review_status(
        st.session_state.scan_id
    )

    st.session_state.review_status = status

    st.divider()
    st.subheader("AI Screening Result")

    prediction = result.get("prediction", "Unknown")
    confidence = float(result.get("confidence", 0))

    c1, c2 = st.columns(2)

    with c1:
        st.metric("AI Finding", prediction)

    with c2:
        st.metric(
            "AI Confidence",
            f"{confidence:.1%}",
        )

    probabilities = result.get("probabilities", {})

    if isinstance(probabilities, dict) and probabilities:

        st.subheader("Probability Breakdown")

        for label, value in probabilities.items():
            value = float(value)

            st.write(
                f"{label}: {value:.1%}"
            )

            st.progress(
                min(max(value, 0), 1)
            )

    # ==========================================================
    # RADIOLOGIST REVIEW
    # ==========================================================

    st.divider()
    st.subheader("Radiologist Review")

    st.info(
        "Every scan must be reviewed and approved by a "
        "qualified radiologist before the final report "
        "can be downloaded."
    )

    if status == "APPROVED":

        st.success(
            "Radiologist review completed and approved."
        )

    elif status == "PENDING":

        st.warning(
            "This examination is awaiting radiologist review."
        )

    else:

        if st.button(
            "Submit for Radiologist Review",
            type="primary",
            use_container_width=True,
        ):

            current_user = user()

            if not current_user:
                st.error("Please log in again.")
                return

            try:

                sb = get_supabase()

                existing = (
                    sb.table("radiologist_requests")
                    .select("id,status")
                    .eq(
                        "scan_id",
                        st.session_state.scan_id,
                    )
                    .limit(1)
                    .execute()
                    .data
                    or []
                )

                if existing:

                    st.session_state.review_status = (
                        "PENDING"
                    )

                else:

                    response = (
                        sb.table("radiologist_requests")
                        .insert({
                            "user_id": current_user.id,
                            "scan_id": st.session_state.scan_id,
                            "status": "PENDING",
                        })
                        .execute()
                    )

                    if not response.data:
                        st.error(
                            "Could not create review request."
                        )
                        return

                    (
                        sb.table("ai_scans")
                        .update({
                            "status":
                                "AWAITING_RADIOLOGIST"
                        })
                        .eq(
                            "id",
                            st.session_state.scan_id,
                        )
                        .execute()
                    )

                    st.session_state.review_status = (
                        "PENDING"
                    )

                st.success(
                    "Scan submitted to the radiologist."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    "Could not submit scan for review."
                )

                st.exception(e)

    # ==========================================================
    # FINAL REPORT
    # ==========================================================

    st.divider()
    st.subheader("Final Medical Report")

    if status != "APPROVED":

        st.error("DOWNLOAD LOCKED")

        st.caption(
            "The final medical report becomes available "
            "only after radiologist approval."
        )

        return

    sb = get_supabase()

    try:

        reports = (
            sb.table("medical_reports")
            .select("report_id,pdf_path")
            .eq(
                "scan_id",
                st.session_state.scan_id,
            )
            .eq(
                "status",
                "APPROVED",
            )
            .order(
                "approved_at",
                desc=True,
            )
            .limit(1)
            .execute()
            .data
            or []
        )

        if not reports:

            st.info(
                "Radiologist approval received. "
                "Final report is being prepared."
            )
            return

        report = reports[0]

        pdf = (
            sb.storage
            .from_("medical-reports")
            .download(report["pdf_path"])
        )

        st.download_button(
            "Download Final Medical Report",
            data=pdf,
            file_name=f"{report['report_id']}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

    except Exception as e:

        st.error(
            "Could not retrieve the approved report."
        )

        st.exception(e)
