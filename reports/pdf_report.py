# ================================================================
# MEDUSA AI
# PDF MEDICAL REPORT GENERATOR
# ================================================================

import io
import uuid
from datetime import datetime

from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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
# SAFE TEXT
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
# SAFE CONFIDENCE
# ================================================================

def format_confidence(value):
    try:
        value = float(value)

        if value > 1:
            value = value / 100

        return f"{value:.1%}"

    except (TypeError, ValueError):
        return "N/A"


# ================================================================
# IMAGE
# ================================================================

def make_report_image(image_bytes, width=150 * mm):

    if not image_bytes:
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90,
        )

        buffer.seek(0)

        if image.width <= 0:
            return None

        ratio = image.height / image.width
        height = width * ratio

        # Prevent extremely large images from breaking the PDF.
        max_height = 220 * mm

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
        "MedusaReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#163A5F"),
        spaceAfter=3 * mm,
    )

    subtitle_style = ParagraphStyle(
        "MedusaReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=4 * mm,
    )

    heading_style = ParagraphStyle(
        "MedusaSectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#163A5F"),
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )

    body_style = ParagraphStyle(
        "MedusaBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        spaceAfter=2 * mm,
    )

    small_style = ParagraphStyle(
        "MedusaSmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555"),
    )

    impression_style = ParagraphStyle(
        "MedusaImpression",
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
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#163A5F"),
        )
    )

    story.append(
        Spacer(1, 4 * mm)
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
        TableStyle(
            [
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
            ]
        )
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
                Paragraph("<b>Patient ID</b>", body_style),
                Paragraph(safe(patient_id), body_style),
            ],
            [
                Paragraph("<b>State</b>", body_style),
                Paragraph(safe(state), body_style),
            ],
        ],
        colWidths=[
            45 * mm,
            135 * mm,
        ],
    )

    patient_table.setStyle(
        TableStyle(
            [
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
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#F4F7FA"),
                ),
            ]
        )
    )

    story.append(patient_table)

    # ============================================================
    # AI SCREENING
    # ============================================================

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
                Paragraph(safe(ai_prediction), body_style),
            ],
            [
                Paragraph("<b>AI Confidence</b>", body_style),
                Paragraph(
                    format_confidence(ai_confidence),
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
        TableStyle(
            [
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
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
            ]
        )
    )

    story.append(ai_table)

    # ============================================================
    # PROBABILITIES
    # ============================================================

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

            try:
                numeric_value = float(value)

                if numeric_value > 1:
                    numeric_value = numeric_value / 100

                formatted_value = f"{numeric_value:.2%}"

            except (TypeError, ValueError):
                formatted_value = "N/A"

            probability_rows.append(
                [
                    Paragraph(
                        safe(name),
                        body_style,
                    ),
                    Paragraph(
                        formatted_value,
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
            TableStyle(
                [
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
                ]
            )
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
            safe(findings) or "Not provided.",
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
            safe(impression) or "Not provided.",
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
                Paragraph("<b>Radiologist</b>", body_style),
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
                Paragraph("<b>Reviewed At</b>", body_style),
                Paragraph(
                    safe(reviewed_at),
                    body_style,
                ),
            ],
            [
                Paragraph("<b>Review Status</b>", body_style),
                Paragraph("APPROVED", body_style),
            ],
        ],
        colWidths=[
            55 * mm,
            125 * mm,
        ],
    )

    radiologist_table.setStyle(
        TableStyle(
            [
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
            ]
        )
    )

    story.append(radiologist_table)

    # ============================================================
    # DISCLAIMER
    # ============================================================

    story.append(
        Spacer(1, 7 * mm)
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor("#D5DDE5"),
        )
    )

    story.append(
        Spacer(1, 3 * mm)
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
        Spacer(1, 4 * mm)
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
