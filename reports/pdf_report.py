# ================================================================
# MEDUSA AI
# WORLD-CLASS MEDICAL PDF REPORT
# ================================================================

from io import BytesIO
from datetime import datetime
from uuid import uuid4

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)
from reportlab.lib.utils import ImageReader


# ================================================================
# REPORT ID
# ================================================================

def generate_report_id():

    date_part = datetime.now().strftime(
        "%Y%m%d"
    )

    unique_part = uuid4().hex[:8].upper()

    return (
        f"MED-R-{date_part}-{unique_part}"
    )


# ================================================================
# SAFE VALUE
# ================================================================

def safe_text(value):

    if value is None:
        return "Not provided"

    value = str(value).strip()

    return value if value else "Not provided"


# ================================================================
# IMAGE
# ================================================================

def make_medical_image(
    image_bytes,
    max_width=175 * mm,
    max_height=95 * mm,
):

    if not image_bytes:
        return None

    try:

        image_reader = ImageReader(
            BytesIO(image_bytes)
        )

        width, height = (
            image_reader.getSize()
        )

        scale = min(
            max_width / width,
            max_height / height,
        )

        image = Image(
            image_reader,
            width=width * scale,
            height=height * scale,
        )

        return image

    except Exception:

        return None


# ================================================================
# PDF GENERATOR
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

    generated_at = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )

    buffer = BytesIO()

    # ============================================================
    # DOCUMENT
    # ============================================================

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=(
            f"Medusa AI Medical Report "
            f"{report_id}"
        ),
        author="Medusa AI",
    )

    # ============================================================
    # STYLES
    # ============================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_LEFT,
        textColor=colors.HexColor(
            "#102A43"
        ),
        spaceAfter=4 * mm,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor(
            "#52606D"
        ),
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor(
            "#102A43"
        ),
        spaceBefore=5 * mm,
        spaceAfter=2.5 * mm,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor(
            "#243B53"
        ),
        spaceAfter=2 * mm,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(
            "#627D98"
        ),
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(
            "#52606D"
        ),
    )

    # ============================================================
    # STORY
    # ============================================================

    story = []

    # ============================================================
    # HEADER
    # ============================================================

    header_data = [
        [
            Paragraph(
                "MEDUSA AI",
                title_style,
            ),
            Paragraph(
                f"<b>FINAL MEDICAL REPORT</b><br/>"
                f"Report ID: {safe_text(report_id)}",
                subtitle_style,
            ),
        ]
    ]

    header = Table(
        header_data,
        colWidths=[
            105 * mm,
            65 * mm,
        ],
    )

    header.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
            (
                "ALIGN",
                (1, 0),
                (1, 0),
                "RIGHT",
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
        ])
    )

    story.append(header)

    story.append(
        Table(
            [[""]],
            colWidths=[174 * mm],
            rowHeights=[1.2 * mm],
            style=TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(
                        "#167D9A"
                    ),
                ),
            ]),
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    # ============================================================
    # PATIENT DETAILS
    # ============================================================

    story.append(
        Paragraph(
            "PATIENT & EXAMINATION DETAILS",
            section_style,
        )
    )

    patient_table_data = [
        [
            Paragraph(
                "<b>Patient Name</b>",
                body_style,
            ),
            Paragraph(
                safe_text(patient_name),
                body_style,
            ),
            Paragraph(
                "<b>Patient ID</b>",
                body_style,
            ),
            Paragraph(
                safe_text(patient_id),
                body_style,
            ),
        ],
        [
            Paragraph(
                "<b>State</b>",
                body_style,
            ),
            Paragraph(
                safe_text(state),
                body_style,
            ),
            Paragraph(
                "<b>Examination</b>",
                body_style,
            ),
            Paragraph(
                safe_text(examination),
                body_style,
            ),
        ],
        [
            Paragraph(
                "<b>Report Date</b>",
                body_style,
            ),
            Paragraph(
                safe_text(generated_at),
                body_style,
            ),
            Paragraph(
                "<b>Report Status</b>",
                body_style,
            ),
            Paragraph(
                "<b>RADIOLOGIST APPROVED</b>",
                body_style,
            ),
        ],
    ]

    patient_table = Table(
        patient_table_data,
        colWidths=[
            30 * mm,
            60 * mm,
            32 * mm,
            52 * mm,
        ],
    )

    patient_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor(
                    "#D9E2EC"
                ),
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#F0F4F8"
                ),
            ),
            (
                "BACKGROUND",
                (2, 0),
                (2, -1),
                colors.HexColor(
                    "#F0F4F8"
                ),
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
        ])
    )

    story.append(
        patient_table
    )

    # ============================================================
    # AI SCREENING
    # ============================================================

    story.append(
        Paragraph(
            "AI-ASSISTED SCREENING",
            section_style,
        )
    )

    ai_confidence = (
        float(ai_confidence) * 100
    )

    ai_data = [
        [
            Paragraph(
                "<b>AI System</b>",
                body_style,
            ),
            Paragraph(
                "Medusa AI",
                body_style,
            ),
        ],
        [
            Paragraph(
                "<b>AI Finding</b>",
                body_style,
            ),
            Paragraph(
                safe_text(ai_prediction),
                body_style,
            ),
        ],
        [
            Paragraph(
                "<b>AI Confidence</b>",
                body_style,
            ),
            Paragraph(
                f"{ai_confidence:.2f}%",
                body_style,
            ),
        ],
    ]

    ai_table = Table(
        ai_data,
        colWidths=[
            45 * mm,
            129 * mm,
        ],
    )

    ai_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor(
                    "#D9E2EC"
                ),
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#F0F4F8"
                ),
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
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
                5,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
        ])
    )

    story.append(ai_table)

    # ============================================================
    # PROBABILITIES
    # ============================================================

    if probabilities:

        story.append(
            Paragraph(
                "AI PROBABILITY BREAKDOWN",
                section_style,
            )
        )

        probability_rows = [
            [
                Paragraph(
                    "<b>Classification</b>",
                    body_style,
                ),
                Paragraph(
                    "<b>Probability</b>",
                    body_style,
                ),
            ]
        ]

        for name, value in probabilities.items():

            probability_rows.append(
                [
                    Paragraph(
                        safe_text(name),
                        body_style,
                    ),
                    Paragraph(
                        f"{float(value) * 100:.2f}%",
                        body_style,
                    ),
                ]
            )

        probability_table = Table(
            probability_rows,
            colWidths=[
                120 * mm,
                54 * mm,
            ],
        )

        probability_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#D9E2EC"
                    ),
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#EAF2F8"
                    ),
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (1, -1),
                    "RIGHT",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
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
                    4,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ])
        )

        story.append(
            probability_table
        )

    # ============================================================
    # MEDICAL IMAGE
    # ============================================================

    medical_image = (
        xray_image
        if xray_image
        else ultrasound_image
    )

    if medical_image:

        story.append(
            Paragraph(
                "EXAMINATION IMAGE",
                section_style,
            )
        )

        image = make_medical_image(
            medical_image
        )

        if image:

            story.append(
                image
            )

            story.append(
                Spacer(
                    1,
                    3 * mm,
                )
            )

    # ============================================================
    # RADIOLOGIST REPORT
    # ============================================================

    story.append(
        Paragraph(
            "RADIOLOGIST REPORT",
            section_style,
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
            safe_text(findings),
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
            safe_text(impression),
            body_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Recommendations</b>",
            body_style,
        )
    )

    story.append(
        Paragraph(
            safe_text(recommendations),
            body_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Radiologist Remarks</b>",
            body_style,
        )
    )

    story.append(
        Paragraph(
            safe_text(remarks),
            body_style,
        )
    )

    # ============================================================
    # RADIOLOGIST AUTHENTICATION
    # ============================================================

    story.append(
        Paragraph(
            "RADIOLOGIST AUTHENTICATION",
            section_style,
        )
    )

    authentication_data = [
        [
            Paragraph(
                "<b>Radiologist</b>",
                body_style,
            ),
            Paragraph(
                safe_text(radiologist_name),
                body_style,
            ),
        ],
        [
            Paragraph(
                "<b>Registration No.</b>",
                body_style,
            ),
            Paragraph(
                safe_text(
                    registration_number
                ),
                body_style,
            ),
        ],
        [
            Paragraph(
                "<b>Reviewed</b>",
                body_style,
            ),
            Paragraph(
                safe_text(reviewed_at),
                body_style,
            ),
        ],
        [
            Paragraph(
                "<b>Status</b>",
                body_style,
            ),
            Paragraph(
                "<b>APPROVED</b>",
                body_style,
            ),
        ],
    ]

    authentication_table = Table(
        authentication_data,
        colWidths=[
            45 * mm,
            129 * mm,
        ],
    )

    authentication_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor(
                    "#D9E2EC"
                ),
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#F0F4F8"
                ),
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
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
                5,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
        ])
    )

    story.append(
        authentication_table
    )

    # ============================================================
    # DISCLAIMER
    # ============================================================

    story.append(
        Spacer(
            1,
            7 * mm,
        )
    )

    disclaimer_box = Table(
        [
            [
                Paragraph(
                    "<b>IMPORTANT MEDICAL NOTICE</b><br/><br/>"
                    "This report contains AI-assisted screening "
                    "information together with the interpretation "
                    "and approval of a qualified reviewing "
                    "radiologist. AI output is not, by itself, "
                    "a medical diagnosis. Clinical findings should "
                    "be interpreted in conjunction with the "
                    "patient's clinical history and other relevant "
                    "investigations.",
                    disclaimer_style,
                )
            ]
        ],
        colWidths=[
            174 * mm
        ],
    )

    disclaimer_box.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor(
                    "#F7F9FC"
                ),
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#BCCCDC"
                ),
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8,
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
        ])
    )

    story.append(
        disclaimer_box
    )

    # ============================================================
    # FOOTER
    # ============================================================

    def add_footer(canvas, document):

        canvas.saveState()

        canvas.setStrokeColor(
            colors.HexColor(
                "#D9E2EC"
            )
        )

        canvas.line(
            18 * mm,
            13 * mm,
            192 * mm,
            13 * mm,
        )

        canvas.setFont(
            "Helvetica",
            7,
        )

        canvas.setFillColor(
            colors.HexColor(
                "#627D98"
            )
        )

        canvas.drawString(
            18 * mm,
            8 * mm,
            f"Medusa AI • Report {report_id}",
        )

        canvas.drawRightString(
            192 * mm,
            8 * mm,
            f"Page {document.page}",
        )

        canvas.restoreState()

    # ============================================================
    # BUILD
    # ============================================================

    document.build(
        story,
        onFirstPage=add_footer,
        onLaterPages=add_footer,
    )

    buffer.seek(0)

    return buffer, report_id
