# ================================================================
# MEDUSA AI
# reports/pdf_report.py
#
# Contains BOTH:
# 1. Final PDF medical report generator
# 2. Medical Reports page for Streamlit
#
# IMPORTANT:
# app.py should import:
#
# from reports.pdf_report import (
#     generate_pdf_report,
#     show_pdf_reports,
# )
# ================================================================

import io
import json
import uuid
from datetime import datetime

import streamlit as st
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
)

from utils.supabase_client import get_supabase


# ================================================================
# REPORT ID
# ================================================================

def generate_report_id():
    return (
        "MED-R-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:8].upper()
    )


# ================================================================
# SAFE TEXT FOR REPORTLAB
# ================================================================

def safe(value):
    if value is None:
        return ""

    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


# ================================================================
# SAFE FLOAT
# ================================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ================================================================
# NORMALIZE CONFIDENCE
# ================================================================

def normalize_confidence(value):
    """
    Handles both:
        0.95  -> 95%
        95    -> 95%
    """

    confidence = safe_float(value)

    if confidence > 1:
        confidence = confidence / 100.0

    return max(0.0, min(confidence, 1.0))


# ================================================================
# IMAGE
# ================================================================

def make_report_image(
    image_bytes,
    width=150 * mm,
):
    if not image_bytes:
        return None

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        original_width, original_height = image.size

        if original_width <= 0 or original_height <= 0:
            return None

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90,
        )

        buffer.seek(0)

        ratio = original_height / original_width
        height = width * ratio

        # Prevent extremely large images from breaking the PDF.
        max_height = 180 * mm

        if height > max_height:
            height = max_height
            width = height / ratio

        return RLImage(
            buffer,
            width=width,
            height=height,
        )

    except Exception:
        return None


# ================================================================
# MAIN PDF GENERATOR
# ================================================================

def generate_pdf_report(
    patient_name,
    patient_id,
    state,
    examination,
    ai_prediction,
    ai_confidence,
    probabilities,
    radiologist_name,
    registration_number,
    findings,
    impression,
    recommendations,
    remarks,
    reviewed_at,
    xray_image=None,
    ultrasound_image=None,
):
    report_id = generate_report_id()

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Medusa AI Medical Report",
        author="Medusa AI",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=5 * mm,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
        textColor=colors.HexColor("#163A5F"),
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        spaceAfter=2 * mm,
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555"),
    )

    impression_style = ParagraphStyle(
        "ReportImpression",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
    )

    story = []

    # ============================================================
    # HEADER
    # ============================================================

    story.append(
        Paragraph(
            "MEDUSA AI",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "AI-ASSISTED MEDICAL IMAGING REPORT",
            subtitle_style,
        )
    )

    story.append(
        Spacer(
            1,
            3 * mm,
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#163A5F"),
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    # ============================================================
    # REPORT IDENTIFICATION
    # ============================================================

    identification = [
        [
            Paragraph("<b>Report ID</b>", body_style),
            Paragraph(safe(report_id), body_style),
            Paragraph("<b>Report Date</b>", body_style),
            Paragraph(
                datetime.now().strftime("%d %B %Y"),
                body_style,
            ),
        ],
        [
            Paragraph("<b>Examination</b>", body_style),
            Paragraph(safe(examination), body_style),
            Paragraph("<b>Patient ID</b>", body_style),
            Paragraph(safe(patient_id), body_style),
        ],
    ]

    identification_table = Table(
        identification,
        colWidths=[
            28 * mm,
            62 * mm,
            28 * mm,
            62 * mm,
        ],
    )

    identification_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#F4F7FA"),
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#D5DDE5"),
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor("#E0E5EA"),
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
        ])
    )

    story.append(identification_table)

    # ============================================================
    # PATIENT INFORMATION
    # ============================================================

    story.append(
        Paragraph(
            "PATIENT INFORMATION",
            heading_style,
        )
    )

    patient_table = Table(
        [
            [
                Paragraph("<b>Patient Name</b>", body_style),
                Paragraph(safe(patient_name), body_style),
            ],
            [
                Paragraph("<b>State</b>", body_style),
                Paragraph(safe(state), body_style),
            ],
            [
                Paragraph("<b>Patient ID</b>", body_style),
                Paragraph(safe(patient_id), body_style),
            ],
        ],
        colWidths=[
            45 * mm,
            135 * mm,
        ],
    )

    patient_table.setStyle(
        TableStyle([
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#D5DDE5"),
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor("#E0E5EA"),
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
        ])
    )

    story.append(patient_table)

    # ============================================================
    # AI SCREENING
    # ============================================================

    confidence = normalize_confidence(ai_confidence)

    story.append(
        Paragraph(
            "AI SCREENING RESULT",
            heading_style,
        )
    )

    ai_table = Table(
        [
            [
                Paragraph("<b>AI System</b>", body_style),
                Paragraph("Medusa AI", body_style),
            ],
            [
                Paragraph("<b>AI Finding</b>", body_style),
                Paragraph(
                    safe(ai_prediction),
                    body_style,
                ),
            ],
            [
                Paragraph("<b>AI Confidence</b>", body_style),
                Paragraph(
                    f"{confidence:.1%}",
                    body_style,
                ),
            ],
        ],
        colWidths=[
            45 * mm,
            135 * mm,
        ],
    )

    ai_table.setStyle(
        TableStyle([
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#D5DDE5"),
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor("#E0E5EA"),
            ),
        ])
    )

    story.append(ai_table)

    # ============================================================
    # PROBABILITIES
    # ============================================================

    if isinstance(probabilities, str):

        try:
            probabilities = json.loads(probabilities)
        except Exception:
            probabilities = {}

    if isinstance(probabilities, dict) and probabilities:

        story.append(
            Paragraph(
                "AI PROBABILITY BREAKDOWN",
                heading_style,
            )
        )

        probability_rows = [
            [
                Paragraph("<b>Class</b>", body_style),
                Paragraph("<b>Probability</b>", body_style),
            ]
        ]

        for name, value in probabilities.items():

            probability = normalize_confidence(value)

            probability_rows.append(
                [
                    Paragraph(
                        safe(name),
                        body_style,
                    ),
                    Paragraph(
                        f"{probability:.2%}",
                        body_style,
                    ),
                ]
            )

        probability_table = Table(
            probability_rows,
            colWidths=[
                100 * mm,
                80 * mm,
            ],
        )

        probability_table.setStyle(
            TableStyle([
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#D5DDE5"),
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.25,
                    colors.HexColor("#E0E5EA"),
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#F4F7FA"),
                ),
            ])
        )

        story.append(probability_table)

    # ============================================================
    # EXAMINATION IMAGE
    # ============================================================

    examination_image = (
        xray_image
        if xray_image
        else ultrasound_image
    )

    report_image = make_report_image(
        examination_image
    )

    if report_image:

        story.append(
            Paragraph(
                "EXAMINATION IMAGE",
                heading_style,
            )
        )

        story.append(report_image)

    # ============================================================
    # RADIOLOGIST REPORT
    # ============================================================

    story.append(
        Paragraph(
            "RADIOLOGIST REPORT",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Findings</b>",
            body_style,
        )
    )

    story.append(
        Paragraph(
            safe(findings),
            body_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Impression</b>",
            body_style,
        )
    )

    story.append(
        Paragraph(
            safe(impression),
            impression_style,
        )
    )

    if recommendations:

        story.append(
            Paragraph(
                "<b>Recommendations</b>",
                body_style,
            )
        )

        story.append(
            Paragraph(
                safe(recommendations),
                body_style,
            )
        )

    if remarks:

        story.append(
            Paragraph(
                "<b>Radiologist Remarks</b>",
                body_style,
            )
        )

        story.append(
            Paragraph(
                safe(remarks),
                body_style,
            )
        )

    # ============================================================
    # RADIOLOGIST AUTHENTICATION
    # ============================================================

    story.append(
        Paragraph(
            "RADIOLOGIST AUTHENTICATION",
            heading_style,
        )
    )

    radiologist_table = Table(
        [
            [
                Paragraph(
                    "<b>Radiologist</b>",
                    body_style,
                ),
                Paragraph(
                    safe(radiologist_name),
                    body_style,
                ),
            ],
            [
                Paragraph(
                    "<b>Registration Number</b>",
                    body_style,
                ),
                Paragraph(
                    safe(registration_number),
                    body_style,
                ),
            ],
            [
                Paragraph(
                    "<b>Reviewed At</b>",
                    body_style,
                ),
                Paragraph(
                    safe(reviewed_at),
                    body_style,
                ),
            ],
            [
                Paragraph(
                    "<b>Review Status</b>",
                    body_style,
                ),
                Paragraph(
                    "APPROVED",
                    body_style,
                ),
            ],
        ],
        colWidths=[
            55 * mm,
            125 * mm,
        ],
    )

    radiologist_table.setStyle(
        TableStyle([
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#D5DDE5"),
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor("#E0E5EA"),
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#F4F7FA"),
            ),
        ])
    )

    story.append(radiologist_table)

    # ============================================================
    # DISCLAIMER
    # ============================================================

    story.append(
        Spacer(
            1,
            7 * mm,
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor("#D5DDE5"),
        )
    )

    story.append(
        Spacer(
            1,
            3 * mm,
        )
    )

    story.append(
        Paragraph(
            "<b>IMPORTANT:</b> This report contains "
            "AI-assisted screening information together "
            "with the professional interpretation of the "
            "reviewing radiologist. The radiologist's "
            "interpretation takes precedence over the AI "
            "screening output. This report should be "
            "interpreted in the appropriate clinical context.",
            small_style,
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    story.append(
        Paragraph(
            f"Report ID: {safe(report_id)}",
            small_style,
        )
    )

    # ============================================================
    # BUILD PDF
    # ============================================================

    document.build(story)

    buffer.seek(0)

    return buffer, report_id


# ################################################################
# ################################################################
#
# MEDICAL REPORTS PAGE
#
# ################################################################
# ################################################################


def _get_current_user():
    """Safely retrieve the currently authenticated Supabase user."""

    try:

        supabase = get_supabase()

        response = (
            supabase
            .auth
            .get_user()
        )

        if response and response.user:
            return response.user

    except Exception:
        pass

    return None


# ================================================================
# DOWNLOAD PDF FROM STORAGE
# ================================================================

def _download_report_pdf(
    supabase,
    pdf_path,
):
    if not pdf_path:
        return None

    try:

        pdf_bytes = (
            supabase
            .storage
            .from_("medical-reports")
            .download(pdf_path)
        )

        return pdf_bytes

    except Exception:
        return None


# ================================================================
# DOWNLOAD EXAMINATION IMAGE
# ================================================================

def _download_scan_image(
    supabase,
    image_path,
):
    if not image_path:
        return None

    try:

        image_bytes = (
            supabase
            .storage
            .from_("mammosense-scans")
            .download(image_path)
        )

        return image_bytes

    except Exception:
        return None


# ================================================================
# MEDICAL REPORTS PAGE
# ================================================================

def show_pdf_reports():

    st.title("Medical Reports")

    st.caption(
        "Radiologist-approved medical reports"
    )

    # ============================================================
    # AUTHENTICATION
    # ============================================================

    user = _get_current_user()

    if user is None:

        st.error(
            "Please log in to view your reports."
        )

        return

    supabase = get_supabase()

    # ============================================================
    # LOAD APPROVED REPORTS
    # ============================================================

    try:

        response = (
            supabase
            .table("medical_reports")
            .select(
                "id,"
                "report_id,"
                "scan_id,"
                "review_id,"
                "user_id,"
                "patient_id,"
                "patient_name,"
                "patient_state,"
                "pdf_path,"
                "status,"
                "approved_at,"
                "created_at"
            )
            .eq(
                "user_id",
                user.id,
            )
            .eq(
                "status",
                "APPROVED",
            )
            .order(
                "approved_at",
                desc=True,
            )
            .execute()
        )

        reports = response.data or []

    except Exception as error:

        st.error(
            "Unable to load medical reports."
        )

        st.exception(error)

        return

    # ============================================================
    # NO REPORTS
    # ============================================================

    if not reports:

        st.info(
            "No radiologist-approved medical reports "
            "are available yet."
        )

        return

    st.success(
        f"{len(reports)} approved report"
        + (
            "s"
            if len(reports) != 1
            else ""
        )
        + " available."
    )

    # ============================================================
    # DISPLAY REPORTS
    # ============================================================

    for report in reports:

        report_database_id = report.get("id")

        report_id = (
            report.get("report_id")
            or report_database_id
            or "Unknown"
        )

        scan_id = report.get("scan_id")

        patient_id = (
            report.get("patient_id")
            or "N/A"
        )

        patient_name = (
            report.get("patient_name")
            or "Unknown"
        )

        patient_state = (
            report.get("patient_state")
            or "N/A"
        )

        pdf_path = report.get("pdf_path")

        approved_at = (
            report.get("approved_at")
            or "N/A"
        )

        review_id = (
            report.get("review_id")
            or "N/A"
        )

        # ========================================================
        # LOAD ASSOCIATED SCAN
        # ========================================================

        scan = {}

        if scan_id:

            try:

                scan_response = (
                    supabase
                    .table("ai_scans")
                    .select(
                        "id,"
                        "user_id,"
                        "patient_id,"
                        "patient_name,"
                        "patient_state,"
                        "examination,"
                        "model,"
                        "prediction,"
                        "confidence,"
                        "probabilities,"
                        "image_path,"
                        "status,"
                        "created_at"
                    )
                    .eq(
                        "id",
                        scan_id,
                    )
                    .limit(1)
                    .execute()
                )

                scan_data = (
                    scan_response.data
                    or []
                )

                if scan_data:
                    scan = scan_data[0]

            except Exception:
                scan = {}

        # ========================================================
        # CARD
        # ========================================================

        with st.container(border=True):

            st.subheader(
                f"Report {report_id}"
            )

            col1, col2 = st.columns(2)

            # ----------------------------------------------------
            # LEFT
            # ----------------------------------------------------

            with col1:

                st.markdown(
                    f"**Patient:** {patient_name}"
                )

                st.markdown(
                    f"**Patient ID:** {patient_id}"
                )

                st.markdown(
                    f"**State:** {patient_state}"
                )

                st.markdown(
                    "**Examination:** "
                    f"{scan.get('examination', 'N/A')}"
                )

            # ----------------------------------------------------
            # RIGHT
            # ----------------------------------------------------

            with col2:

                st.markdown(
                    f"**Report ID:** {report_id}"
                )

                st.markdown(
                    f"**Review ID:** {review_id}"
                )

                st.markdown(
                    f"**Approved:** {approved_at}"
                )

                st.markdown(
                    "**Status:** APPROVED"
                )

            # ====================================================
            # SCAN INFORMATION
            # ====================================================

            if scan:

                st.divider()

                st.markdown(
                    "### Examination"
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.write(
                        "**AI Finding:** "
                        f"{scan.get('prediction', 'N/A')}"
                    )

                with c2:

                    raw_confidence = scan.get(
                        "confidence"
                    )

                    if raw_confidence is not None:

                        confidence = (
                            normalize_confidence(
                                raw_confidence
                            )
                        )

                        st.write(
                            "**AI Confidence:** "
                            f"{confidence:.1%}"
                        )

                    else:

                        st.write(
                            "**AI Confidence:** N/A"
                        )

                with c3:

                    st.write(
                        "**Model:** "
                        f"{scan.get('model', 'N/A')}"
                    )

                # =================================================
                # PROBABILITIES
                # =================================================

                probabilities = scan.get(
                    "probabilities",
                    {},
                )

                if isinstance(
                    probabilities,
                    str,
                ):

                    try:
                        probabilities = json.loads(
                            probabilities
                        )
                    except Exception:
                        probabilities = {}

                if (
                    isinstance(
                        probabilities,
                        dict,
                    )
                    and probabilities
                ):

                    with st.expander(
                        "AI Probability Distribution"
                    ):

                        for (
                            label,
                            value,
                        ) in probabilities.items():

                            probability = (
                                normalize_confidence(
                                    value
                                )
                            )

                            st.write(
                                f"**{label}:** "
                                f"{probability:.2%}"
                            )

                            st.progress(
                                probability
                            )

                # =================================================
                # ORIGINAL IMAGE
                # =================================================

                image_path = scan.get(
                    "image_path"
                )

                if image_path:

                    image_bytes = (
                        _download_scan_image(
                            supabase,
                            image_path,
                        )
                    )

                    if image_bytes:

                        st.image(
                            image_bytes,
                            caption=(
                                scan.get(
                                    "examination",
                                    "Examination Image",
                                )
                            ),
                            use_container_width=True,
                        )

                    else:

                        st.warning(
                            "The examination image "
                            "could not be loaded."
                        )

            # ====================================================
            # FINAL PDF
            # ====================================================

            st.divider()

            if not pdf_path:

                st.warning(
                    "This report is approved, but "
                    "its PDF file has not been stored yet."
                )

                continue

            pdf_bytes = _download_report_pdf(
                supabase,
                pdf_path,
            )

            if not pdf_bytes:

                st.error(
                    "The report exists, but its PDF "
                    "could not be loaded from storage."
                )

                continue

            st.success(
                "Radiologist-approved report available."
            )

            st.download_button(
                label="Download Final Medical Report",
                data=pdf_bytes,
                file_name=f"{report_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key=(
                    f"download_report_"
                    f"{report_database_id}"
                ),
            )
