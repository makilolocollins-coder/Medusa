# ================================================================
# MEDUSA AI
# PROFESSIONAL MEDICAL PDF REPORT
#
# PDF IS ONLY CREATED AFTER RADIOLOGIST APPROVAL.
# ================================================================

import io
import uuid
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
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
    Image as RLImage,
    KeepTogether,
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


# ================================================================
# REPORT ID
# ================================================================

def generate_report_id():

    timestamp = datetime.now().strftime(
        "%Y%m%d%H%M%S"
    )

    random_part = uuid.uuid4().hex[:6].upper()

    return (
        f"MED-RPT-{timestamp}-{random_part}"
    )


# ================================================================
# SAFE TEXT
# ================================================================

def clean_text(value):

    if value is None:
        return ""

    return str(value).strip()


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
    recommendations="",
    remarks="",
    reviewed_at=None,
    xray_image=None,
    ultrasound_image=None,
):

    report_id = generate_report_id()

    # ============================================================
    # PDF BUFFER
    # ============================================================

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

    # ============================================================
    # STYLES
    # ============================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        alignment=TA_LEFT,
        spaceAfter=4 * mm,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor(
            "#5B6573"
        ),
        leading=13,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor(
            "#17324D"
        ),
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=2 * mm,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(
            "#667085"
        ),
    )

    impression_style = ParagraphStyle(
        "Impression",
        parent=body_style,
        fontSize=10,
        leading=15,
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
                "<b>MEDUSA AI</b>",
                title_style,
            ),
            Paragraph(
                "<b>MEDICAL IMAGING REPORT</b><br/>"
                "AI-assisted screening with "
                "radiologist review",
                subtitle_style,
            ),
        ]
    ]

    header = Table(
        header_data,
        colWidths=[
            75 * mm,
            90 * mm,
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
                8,
            ),
            (
                "LINEBELOW",
                (0, 0),
                (-1, -1),
                1,
                colors.HexColor("#17324D"),
            ),
        ])
    )

    story.append(header)

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    # ============================================================
    # REPORT METADATA
    # ============================================================

    report_date = (
        reviewed_at
        if reviewed_at
        else datetime.now().strftime(
            "%d %B %Y, %H:%M"
        )
    )

    metadata = [
        [
            Paragraph(
                "<b>Report ID</b>",
                body_style,
            ),
            Paragraph(
                clean_text(report_id),
                body_style,
            ),
            Paragraph(
                "<b>Report Date</b>",
                body_style,
            ),
            Paragraph(
                clean_text(report_date),
                body_style,
            ),
        ],
        [
            Paragraph(
                "<b>Examination</b>",
                body_style,
            ),
            Paragraph(
                clean_text(examination),
                body_style,
            ),
            Paragraph(
                "<b>Patient ID</b>",
                body_style,
            ),
            Paragraph(
                clean_text(patient_id),
                body_style,
            ),
        ],
    ]

    metadata_table = Table(
        metadata,
        colWidths=[
            28 * mm,
            57 * mm,
            30 * mm,
            50 * mm,
        ],
    )

    metadata_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#F6F8FA"),
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#D0D5DD"),
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.3,
                colors.HexColor("#E4E7EC"),
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

    story.append(metadata_table)

    # ============================================================
    # PATIENT
    # ============================================================

    story.append(
        Paragraph(
            "PATIENT INFORMATION",
            section_style,
        )
    )

    patient_table = Table(
        [
            [
                Paragraph(
                    "<b>Patient Name</b>",
                    body_style,
                ),
                Paragraph(
                    clean_text(patient_name),
                    body_style,
                ),
            ],
            [
                Paragraph(
                    "<b>State</b>",
                    body_style,
                ),
                Paragraph(
                    clean_text(state),
                    body_style,
                ),
            ],
            [
                Paragraph(
                    "<b>Patient ID</b>",
                    body_style,
                ),
                Paragraph(
                    clean_text(patient_id),
                    body_style,
                ),
            ],
        ],
        colWidths=[
            45 * mm,
            120 * mm,
        ],
    )

    patient_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#D0D5DD"),
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#F9FAFB"),
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

    story.append(patient_table)

    # ============================================================
    # IMAGE
    # ============================================================

    selected_image = (
        xray_image
        if xray_image is not None
        else ultrasound_image
    )

    if selected_image:

        try:

            image_buffer = io.BytesIO(
                selected_image
            )

            medical_image = RLImage(
                image_buffer,
                width=105 * mm,
                height=75 * mm,
                kind="proportional",
            )

            story.append(
                Paragraph(
                    "EXAMINATION IMAGE",
                    section_style,
                )
            )

            image_table = Table(
                [[medical_image]],
                colWidths=[
                    165 * mm
                ],
            )

            image_table.setStyle(
                TableStyle([
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER",
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#D0D5DD"),
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                ])
            )

            story.append(image_table)

        except Exception:
            pass

    # ============================================================
    # AI RESULT
    # ============================================================

    story.append(
        Paragraph(
            "AI SCREENING RESULT",
            section_style,
        )
    )

    ai_confidence_percent = (
        float(ai_confidence) * 100
    )

    ai_table = Table(
        [
            [
                Paragraph(
                    "<b>AI Model</b>",
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
                    clean_text(ai_prediction),
                    body_style,
                ),
            ],
            [
                Paragraph(
                    "<b>AI Confidence</b>",
                    body_style,
                ),
                Paragraph(
                    f"{ai_confidence_percent:.1f}%",
                    body_style,
                ),
            ],
        ],
        colWidths=[
            45 * mm,
            120 * mm,
        ],
    )

    ai_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#D0D5DD"),
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#F9FAFB"),
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

    if isinstance(probabilities, dict):

        probability_rows = [
            [
                Paragraph(
                    "<b>Class</b>",
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
                        clean_text(name),
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
                100 * mm,
                65 * mm,
            ],
        )

        probability_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D0D5DD"),
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#EEF2F6"),
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (1, -1),
                    "RIGHT",
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

        story.append(probability_table)

    # ============================================================
    # RADIOLOGIST REVIEW
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
            clean_text(findings).replace(
                "\n",
                "<br/>",
            ),
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
            clean_text(impression).replace(
                "\n",
                "<br/>",
            ),
            impression_style,
        )
    )

    if clean_text(recommendations):

        story.append(
            Paragraph(
                "<b>Recommendations</b>",
                body_style,
            )
        )

        story.append(
            Paragraph(
                clean_text(
                    recommendations
                ).replace(
                    "\n",
                    "<br/>",
                ),
                body_style,
            )
        )

    if clean_text(remarks):

        story.append(
            Paragraph(
                "<b>Radiologist Remarks</b>",
                body_style,
            )
        )

        story.append(
            Paragraph(
                clean_text(
                    remarks
                ).replace(
                    "\n",
                    "<br/>",
                ),
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

    radiologist_table = Table(
        [
            [
                Paragraph(
                    "<b>Radiologist</b>",
                    body_style,
                ),
                Paragraph(
                    clean_text(
                        radiologist_name
                    ),
                    body_style,
                ),
            ],
            [
                Paragraph(
                    "<b>Registration No.</b>",
                    body_style,
                ),
                Paragraph(
                    clean_text(
                        registration_number
                    ),
                    body_style,
                ),
            ],
            [
                Paragraph(
                    "<b>Review Status</b>",
                    body_style,
                ),
                Paragraph(
                    "RADIOLOGIST APPROVED",
                    body_style,
                ),
            ],
            [
                Paragraph(
                    "<b>Reviewed</b>",
                    body_style,
                ),
                Paragraph(
                    clean_text(
                        report_date
                    ),
                    body_style,
                ),
            ],
        ],
        colWidths=[
            45 * mm,
            120 * mm,
        ],
    )

    radiologist_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#D0D5DD"),
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#F9FAFB"),
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

    story.append(radiologist_table)

    # ============================================================
    # DISCLAIMER
    # ============================================================

    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )

    disclaimer = (
        "<b>Important:</b> This report contains "
        "AI-assisted screening information and "
        "the final clinical interpretation of a "
        "qualified radiologist. AI output should "
        "not be interpreted independently as a "
        "medical diagnosis."
    )

    story.append(
        Paragraph(
            disclaimer,
            small_style,
        )
    )

    # ============================================================
    # FOOTER
    # ============================================================

    def footer(canvas, doc):

        canvas.saveState()

        canvas.setFont(
            "Helvetica",
            7,
        )

        canvas.setFillColor(
            colors.HexColor("#667085")
        )

        canvas.drawString(
            18 * mm,
            10 * mm,
            f"Medusa AI • Report {report_id}",
        )

        canvas.drawRightString(
            A4[0] - 18 * mm,
            10 * mm,
            f"Page {doc.page}",
        )

        canvas.restoreState()

    # ============================================================
    # BUILD
    # ============================================================

    document.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
    )

    buffer.seek(0)

    return buffer, report_id
