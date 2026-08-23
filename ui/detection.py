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

from ai.tuberculosis import (
load_model as load_tb_model,
predict as predict_tb,
)

STATES = [
"Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi",
"Bayelsa", "Benue", "Borno", "Cross River", "Delta",
"Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo",
"Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi",
"Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo",
"Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba",
"Yobe", "Zamfara", "FCT",
]

def init_state():

defaults = {
    "patient_id": None,
    "scan_id": None,
    "scan_result": None,
    "scan_image_bytes": None,
    "scan_filename": None,
    "review_status": "NOT_REQUESTED",
    "review_id": None,
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value

def current_user():

try:

    response = get_supabase().auth.get_user()

    return response.user

except Exception:

    return None

def new_patient_id():

return (
    "MED-P-"
    + datetime.now().strftime("%Y%m%d")
    + "-"
    + uuid.uuid4().hex[:8].upper()
)

def reset_scan():

for key in [
    "scan_id",
    "scan_result",
    "scan_image_bytes",
    "scan_filename",
    "review_id",
]:

    st.session_state[key] = None

st.session_state.review_status = "NOT_REQUESTED"

def review_status(scan_id):

if not scan_id:

    return "NOT_REQUESTED", None

sb = get_supabase()

try:

    requests = (
        sb.table("radiologist_requests")
        .select("id,status")
        .eq("scan_id", scan_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )

    reviews = (
        sb.table("radiologist_reviews")
        .select("id,status,approved")
        .eq("scan_id", scan_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )

    if reviews:

        r = reviews[0]

        if (
            str(
                r.get("status", "")
            ).upper()
            == "APPROVED"
            or r.get("approved") is True
        ):

            return "APPROVED", r.get("id")

    if requests:

        status = str(
            requests[0].get(
                "status",
                "",
            )
        ).upper()

        if status in (
            "PENDING",
            "REQUESTED",
        ):

            return "PENDING", None

        if status in (
            "APPROVED",
            "COMPLETED",
            "REVIEWED",
        ):

            return "APPROVED", None

    return "NOT_REQUESTED", None

except Exception:

    return (
        st.session_state.get(
            "review_status",
            "NOT_REQUESTED",
        ),
        st.session_state.get(
            "review_id"
        ),
    )

def show_detection():

init_state()

set_background("detection.jpg")

st.title("Medusa AI")

st.caption(
    "AI-assisted medical imaging and radiologist review"
)

# ==========================================================
# PATIENT REGISTRATION
# ==========================================================

st.subheader("Patient Registration")

c1, c2 = st.columns(2)

with c1:

    patient_name = st.text_input(
        "Patient full name",
        placeholder="Enter patient's full name",
        key="patient_name_input",
    ).strip()

with c2:

    patient_state = st.selectbox(
        "State",
        STATES,
        index=STATES.index("Delta"),
        key="patient_state_input",
    )

if (
    patient_name
    and not st.session_state.patient_id
):

    st.session_state.patient_id = (
        new_patient_id()
    )

if st.session_state.patient_id:

    st.info(
        f"Patient ID: "
        f"{st.session_state.patient_id}"
    )

# ==========================================================
# NEW EXAMINATION
# ==========================================================

if st.session_state.scan_result is not None:

    if st.button(
        "Start New Examination",
        use_container_width=True,
    ):

        reset_scan()

        st.rerun()

# ==========================================================
# EXAMINATION TYPE
# ==========================================================

st.subheader("Examination")

model_choice = st.selectbox(
    "AI model",
    [
        "MammoSense — Breast Ultrasound",
        "MammoSense Pneumonia — Chest X-ray",
        "MammoSense TB — Chest X-ray",
    ],
    key="medical_model_input",
)

# ==========================================================
# MODEL SELECTION
# ==========================================================

mammography = (
    model_choice
    == "MammoSense — Breast Ultrasound"
)

pneumonia = (
    model_choice
    == "MammoSense Pneumonia — Chest X-ray"
)

tuberculosis = (
    model_choice
    == "MammoSense TB — Chest X-ray"
)

if mammography:

    examination = "Breast Ultrasound"

    model_name = "MammoSense V2"

elif pneumonia:

    examination = "Chest X-ray"

    model_name = "MammoSense Pneumonia V2"

else:

    examination = "Chest X-ray"

    model_name = "MammoSense TB V12"

# ==========================================================
# UPLOAD INSTRUCTIONS
# ==========================================================

if tuberculosis:

    st.info(
        "Upload a chest X-ray for "
        "AI-assisted tuberculosis screening."
    )

elif pneumonia:

    st.info(
        "Upload a chest X-ray for "
        "AI-assisted pneumonia screening."
    )

else:

    st.info(
        "Upload a breast ultrasound for "
        "AI-assisted breast lesion screening."
    )

# ==========================================================
# IMAGE
# ==========================================================

uploaded = st.file_uploader(
    "Upload medical image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    key="medical_scan_upload",
)

if uploaded:

    image_bytes = uploaded.getvalue()

    st.session_state.scan_image_bytes = (
        image_bytes
    )

    st.session_state.scan_filename = (
        uploaded.name
    )

else:

    image_bytes = (
        st.session_state.scan_image_bytes
    )

if not image_bytes:

    st.warning(
        "Upload an image to continue."
    )

    return

# ==========================================================
# DISPLAY IMAGE
# ==========================================================

try:

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    st.image(
        image,
        caption=examination,
        use_container_width=True,
    )

except Exception as error:

    st.error(
        "The uploaded image could not be opened."
    )

    st.exception(error)

    return

# ==========================================================
# AI ANALYSIS
# ==========================================================

if st.button(
    "Analyze Examination",
    type="primary",
    use_container_width=True,
    key="analyze_medical_scan",
):

    if not patient_name:

        st.error(
            "Enter the patient's full name first."
        )

        return

    user = current_user()

    if not user:

        st.error(
            "Your login session has expired."
        )

        return

    try:

        with st.spinner(
            "Medusa AI is analyzing..."
        ):

            # ------------------------------------------------
            # BREAST ULTRASOUND
            # ------------------------------------------------

            if mammography:

                load_mammo_model()

                result = predict_mammo(
                    image
                )

            # ------------------------------------------------
            # PNEUMONIA
            # ------------------------------------------------

            elif pneumonia:

                load_pneumonia_model()

                result = predict_pneumonia(
                    image
                )

            # ------------------------------------------------
            # TUBERCULOSIS
            # ------------------------------------------------

            elif tuberculosis:

                load_tb_model()

                result = predict_tb(
                    image
                )

            else:

                raise RuntimeError(
                    "No valid AI model selected."
                )

        # ====================================================
        # SAVE IMAGE
        # ====================================================

        sb = get_supabase()

        extension = (
            uploaded.name.rsplit(
                ".",
                1,
            )[-1].lower()
            if uploaded
            else "png"
        )

        content_type = (
            uploaded.type
            if uploaded
            else "image/png"
        )

        image_path = (
            f"{user.id}/"
            f"{st.session_state.patient_id}/"
            f"{uuid.uuid4().hex}."
            f"{extension}"
        )

        sb.storage.from_(
            "mammosense-scans"
        ).upload(
            image_path,
            image_bytes,
            {
                "content-type":
                    content_type,
                "upsert":
                    "false",
            },
        )

        # ====================================================
        # SAVE AI SCAN
        # ====================================================

        response = (
            sb.table("ai_scans")
            .insert({
                "user_id":
                    user.id,

                "patient_id":
                    st.session_state.patient_id,

                "patient_name":
                    patient_name,

                "patient_state":
                    patient_state,

                "examination":
                    examination,

                "model":
                    model_name,

                "prediction":
                    result["prediction"],

                "confidence":
                    result["confidence"],

                "probabilities":
                    result.get(
                        "probabilities",
                        {},
                    ),

                "image_path":
                    image_path,

                "status":
                    "AI_COMPLETED",
            })
            .execute()
        )

        if not response.data:

            st.error(
                "The scan could not be saved."
            )

            return

        st.session_state.scan_id = (
            response.data[0]["id"]
        )

        st.session_state.scan_result = (
            result
        )

        st.session_state.review_status = (
            "NOT_REQUESTED"
        )

        st.session_state.review_id = None

        st.success(
            "AI analysis completed."
        )

        st.rerun()

    except Exception as error:

        st.error(
            "The examination could not be analyzed."
        )

        st.exception(error)

        return

# ==========================================================
# RESULT
# ==========================================================

result = st.session_state.scan_result

if not result:

    return

status, review_id = review_status(
    st.session_state.scan_id
)

st.session_state.review_status = status

st.session_state.review_id = review_id

st.divider()

st.subheader(
    "AI Screening Result"
)

prediction = str(
    result.get(
        "prediction",
        "Unknown",
    )
)

confidence = float(
    result.get(
        "confidence",
        0,
    )
)

c1, c2 = st.columns(2)

with c1:

    dangerous_findings = (
        "MALIGNANT",
        "PNEUMONIA",
        "TB",
        "TUBERCULOSIS",
    )

    if prediction.upper() in (
        dangerous_findings
    ):

        st.error(
            f"Finding: {prediction}"
        )

    else:

        st.success(
            f"Finding: {prediction}"
        )

with c2:

    st.metric(
        "AI Confidence",
        f"{confidence:.1%}",
    )

# ==========================================================
# TB-SPECIFIC RESULT
# ==========================================================

if tuberculosis:

    if prediction.upper() == "TB":

        st.error(
            "TB DETECTED BY AI SCREENING"
        )

        st.warning(
            "This is an AI-assisted screening "
            "result and does not establish a "
            "definitive tuberculosis diagnosis."
        )

    else:

        st.success(
            "NON-TB classification by AI screening."
        )

        st.warning(
            "A NON-TB result does not completely "
            "exclude tuberculosis. Clinical "
            "assessment remains necessary."
        )

# ==========================================================
# PROBABILITIES
# ==========================================================

probabilities = result.get(
    "probabilities",
    {},
)

if (
    isinstance(
        probabilities,
        dict,
    )
    and probabilities
):

    st.subheader(
        "Probability Breakdown"
    )

    for label, value in probabilities.items():

        value = float(value)

        st.write(
            f"{label}: {value:.2%}"
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

# ==========================================================
# RADIOLOGIST REVIEW
# ==========================================================

st.divider()

st.subheader(
    "Radiologist Review"
)

st.info(
    "Every examination must be reviewed and approved "
    "by a qualified radiologist before the final "
    "medical report can be downloaded."
)

if status == "APPROVED":

    st.success(
        "Radiologist review completed and approved."
    )

elif status == "PENDING":

    st.warning(
        "Examination is waiting for radiologist review."
    )

else:

    st.warning(
        "Radiologist review has not been requested."
    )

    if st.button(
        "Submit for Radiologist Review",
        type="primary",
        use_container_width=True,
        key="submit_radiologist_request",
    ):

        user = current_user()

        if not user:

            st.error(
                "Please log in again."
            )

            return

        try:

            sb = get_supabase()

            scan_id = (
                st.session_state.scan_id
            )

            existing = (
                sb.table(
                    "radiologist_requests"
                )
                .select(
                    "id,status"
                )
                .eq(
                    "scan_id",
                    scan_id,
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

                st.info(
                    "This examination is already "
                    "waiting for review."
                )

                st.rerun()

            request = (
                sb.table(
                    "radiologist_requests"
                )
                .insert({
                    "user_id":
                        user.id,

                    "scan_id":
                        scan_id,

                    "status":
                        "PENDING",
                })
                .execute()
            )

            if not request.data:

                st.error(
                    "The review request "
                    "could not be created."
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
                    scan_id,
                )
                .execute()
            )

            st.session_state.review_status = (
                "PENDING"
            )

            st.success(
                "Examination successfully submitted "
                "for radiologist review."
            )

            st.rerun()

        except Exception as error:

            st.error(
                "Could not submit the examination "
                "for review."
            )

            st.exception(error)

# ==========================================================
# FINAL REPORT
# ==========================================================

st.divider()

st.subheader(
    "Final Medical Report"
)

if status != "APPROVED":

    st.error(
        "DOWNLOAD LOCKED"
    )

    st.caption(
        "The final medical report becomes available "
        "only after radiologist review and approval."
    )

    return

st.success(
    "Radiologist-approved report available."
)

try:

    sb = get_supabase()

    reports = (
        sb.table(
            "medical_reports"
        )
        .select(
            "report_id,pdf_path,status"
        )
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
            "The examination has been approved. "
            "The final report is being prepared."
        )

        return

    report = reports[0]

    pdf_path = report.get(
        "pdf_path"
    )

    report_id = report.get(
        "report_id",
        "MEDUSA_REPORT",
    )

    if not pdf_path:

        st.warning(
            "The report exists but its PDF file "
            "is not available yet."
        )

        return

    pdf = (
        sb.storage
        .from_(
            "medical-reports"
        )
        .download(
            pdf_path
        )
    )

    st.download_button(
        "Download Final Medical Report",
        data=pdf,
        file_name=f"{report_id}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
        key=(
            f"download_detection_report_"
            f"{report_id}"
        ),
    )

except Exception as error:

    st.error(
        "The approved report could not be loaded."
    )

    st.exception(error)

st.divider()

st.caption(
    "Medusa AI provides AI-assisted screening and "
    "does not replace professional medical diagnosis "
    "or treatment."
)
