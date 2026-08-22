# ================================================================
# MEDUSA AI
# WORLD-CLASS RADIOLOGY PDF REPORT GENERATOR
#
# PDF DOWNLOAD IS ONLY ALLOWED AFTER RADIOLOGIST APPROVAL
# ================================================================

from io import BytesIO
from datetime import datetime

import qrcode

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
    KeepTogether,
)


# ================================================================
# COLORS
# ================================================================

NAVY = colors.HexColor("#12263A")
BLUE = colors.HexColor("#1769AA")
LIGHT_BLUE = colors.HexColor("#EAF4FB")
LIGHT_GRAY = colors.HexColor("#F4F6F8")
MID_GRAY = colors.HexColor("#6B7280")
DARK = colors.HexColor("#17202A")
GREEN = colors.HexColor("#087F5B")
RED = colors.HexColor("#C92A2A")
WHITE = colors.white


# ================================================================
# REPORT ID
# ================================================================

def generate_report_id(patient_id):
    """
    Generates a unique report ID for each examination.
    """

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    return f"MED-R-{timestamp}-{patient_id[-5:]}"


# ================================================================
# QR CODE
# ================================================================

def create_qr(report_id):

    qr = qrcode.QRCode(
        version=1,
        box_size=4,
        border=2,
    )

    qr.add_data(
        f"MEDUSA REPORT {report_id}"
    )

    qr.make(
        fit=True
    )

    img = qr.make_image(
        fill_color="black",
        back_color="white",
    )

    buffer = BytesIO()

    img.save(
        buffer,
        format="PNG",
    )

    buffer.seek(0)

    return buffer


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
    heatmap_image=None,
):
    """
    Generates a FINAL radiology report.

    IMPORTANT:
    This function should only be called after
    radiologist approval.
    """

    # ------------------------------------------------------------
    # SECURITY CHECK
    # ------------------------------------------------------------

    if not radiologist_name.strip():
        raise ValueError(
            "Radiologist name is required."
        )

    if not findings.strip():
        raise ValueError(
            "Radiologist findings are required."
        )

    if not impression.strip():
        raise ValueError(
            "Radiologist impression is required."
        )

    # ------------------------------------------------------------
    # REPORT ID
    # ------------------------------------------------------------

    report_id = generate_report_id(
        patient_id
    )

    # ------------------------------------------------------------
    # PDF BUFFER
    # ------------------------------------------------------------

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,
        title=f"Medusa Radiology Report {report_id}",
        author="Medusa AI",
    )

    # ------------------------------------------------------------
    # STYLES
    # ------------------------------------------------------------

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=NAVY,
        alignment=TA_LEFT,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=MID_GRAY,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=WHITE,
        spaceBefore=7,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=14,
        textColor=DARK,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=MID_GRAY,
    )

    result_style = ParagraphStyle(
        "Result",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=NAVY,
    )

    # ------------------------------------------------------------
    # STORY
    # ------------------------------------------------------------

    story = []

    # ============================================================
    # HEADER
    # ============================================================

    header = Table(
        [
            [
                Paragraph(
                    "MEDUSA◉",
                    title_style,
                ),
                Paragraph(
                    "<b>RADIOLOGY REPORT</b><br/>"
                    "AI-ASSISTED MEDICAL IMAGING",
                    subtitle_style,
                ),
            ]
        ],
        colWidths=[
            90 * mm,
            85 * mm,
        ],
    )

    header.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
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
            ]
        )
    )

    story.append(
        header
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1.2,
            color=NAVY,
            spaceAfter=10,
        )
    )

    # ============================================================
    # PATIENT INFORMATION
    # ============================================================

    story.append(
        Table(
            [
                [
                    Paragraph(
                        "PATIENT INFORMATION",
                        section_style,
                    )
                ]
            ],
            colWidths=[175 * mm],
            style=TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        NAVY,
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        8,
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
                ]
            ),
        )
    )

    examination_date = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )

    patient_data = [
        [
            Paragraph("<b>Patient Name</b>", body_style),
            Paragraph(patient_name, body_style),
            Paragraph("<b>Patient ID</b>", body_style),
            Paragraph(patient_id, body_style),
        ],
        [
            Paragraph("<b>State</b>", body_style),
            Paragraph(state, body_style),
            Paragraph("<b>Report ID</b>", body_style),
            Paragraph(report_id, body_style),
        ],
        [
            Paragraph("<b>Examination</b>", body_style),
            Paragraph(examination, body_style),
            Paragraph("<b>Date</b>", body_style),
            Paragraph(examination_date, body_style),
        ],
    ]

    patient_table = Table(
        patient_data,
        colWidths=[
            30 * mm,
            60 * mm,
            30 * mm,
            55 * mm,
        ],
    )

    patient_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_GRAY,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor("#D7DCE1"),
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
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
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
            ]
        )
    )

    story.append(
        patient_table
    )

    # ============================================================
    # X-RAY IMAGE
    # ============================================================

    if xray_image is not None:

        story.append(
            Spacer(
                1,
                8,
            )
        )

        story.append(
            Paragraph(
                "EXAMINATION IMAGE",
                section_style,
            )
        )

        try:

            xray = RLImage(
                xray_image,
                width=95 * mm,
                height=95 * mm,
                kind="proportional",
            )

            story.append(
                xray
            )

        except Exception:
            pass

    # ============================================================
    # AI SCREENING
    # ============================================================

    story.append(
        Spacer(
            1,
            8,
        )
    )

    story.append(
        Paragraph(
            "AI SCREENING RESULT",
            section_style,
        )
    )

    prediction_text = ai_prediction.upper()

    confidence_percent = (
        float(ai_confidence) * 100
    )

    prediction_color = (
        RED
        if prediction_text == "PNEUMONIA"
        else GREEN
    )

    result_table = Table(
        [
            [
                Paragraph(
                    "AI PREDICTION",
                    small_style,
                ),
                Paragraph(
                    prediction_text,
                    ParagraphStyle(
                        "Prediction",
                        parent=result_style,
                        textColor=prediction_color,
                    ),
                ),
                Paragraph(
                    f"{confidence_percent:.1f}%",
                    result_style,
                ),
            ]
        ],
        colWidths=[
            45 * mm,
            75 * mm,
            55 * mm,
        ],
    )

    result_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_BLUE,
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
                    9,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
            ]
        )
    )

    story.append(
        result_table
    )

    # ============================================================
    # PROBABILITIES
    # ============================================================

    normal = float(
        probabilities.get(
            "NORMAL",
            0,
        )
    ) * 100

    pneumonia = float(
        probabilities.get(
            "PNEUMONIA",
            0,
        )
    ) * 100

    probability_table = Table(
        [
            [
                Paragraph(
                    "<b>CLASS</b>",
                    small_style,
                ),
                Paragraph(
                    "<b>PROBABILITY</b>",
                    small_style,
                ),
            ],
            [
                Paragraph(
                    "Normal",
                    body_style,
                ),
                Paragraph(
                    f"{normal:.2f}%",
                    body_style,
                ),
            ],
            [
                Paragraph(
                    "Pneumonia",
                    body_style,
                ),
                Paragraph(
                    f"{pneumonia:.2f}%",
                    body_style,
                ),
            ],
        ],
        colWidths=[
            120 * mm,
            55 * mm,
        ],
    )

    probability_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor("#D7DCE1"),
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    LIGHT_GRAY,
                ),
                (
                    "ALIGN",
                    (1, 0),
                    (1, -1),
                    "RIGHT",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
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

    story.append(
        probability_table
    )

    # ============================================================
    # RADIOLOGIST REVIEW
    # ============================================================

    story.append(
        Spacer(
            1,
            8,
        )
    )

    story.append(
        Paragraph(
            "RADIOLOGIST REVIEW",
            section_style,
        )
    )

    review_sections = [
        (
            "FINDINGS",
            findings,
        ),
        (
            "IMPRESSION",
            impression,
        ),
        (
            "RECOMMENDATIONS",
            recommendations
            if recommendations.strip()
            else "None provided.",
        ),
        (
            "RADIOLOGIST REMARKS",
            remarks
            if remarks.strip()
            else "None provided.",
        ),
    ]

    for heading, text in review_sections:

        block = Table(
            [
                [
                    Paragraph(
                        f"<b>{heading}</b>",
                        body_style,
                    )
                ],
                [
                    Paragraph(
                        text.replace(
                            "\n",
                            "<br/>",
                        ),
                        body_style,
                    )
                ],
            ],
            colWidths=[
                175 * mm
            ],
        )

        block.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        LIGHT_GRAY,
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor(
                            "#D7DCE1"
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
                ]
            )
        )

        story.append(
            block
        )

        story.append(
            Spacer(
                1,
                4,
            )
        )

    # ============================================================
    # FINAL AUTHORIZATION
    # ============================================================

    story.append(
        Spacer(
            1,
            8,
        )
    )

    story.append(
        Paragraph(
            "FINAL AUTHORIZATION",
            section_style,
        )
    )

    authorization = Table(
        [
            [
                Paragraph(
                    "<b>✓ RADIOLOGIST REVIEWED AND APPROVED</b>",
                    ParagraphStyle(
                        "Approved",
                        parent=body_style,
                        textColor=GREEN,
                        fontSize=11,
                    ),
                )
            ],
            [
                Paragraph(
                    f"<b>Radiologist:</b> {radiologist_name}",
                    body_style,
                )
            ],
            [
                Paragraph(
                    f"<b>Registration No.:</b> "
                    f"{registration_number}",
                    body_style,
                )
            ],
            [
                Paragraph(
                    f"<b>Reviewed:</b> {reviewed_at}",
                    body_style,
                )
            ],
            [
                Paragraph(
                    "<b>Report Status:</b> FINAL",
                    body_style,
                )
            ],
        ],
        colWidths=[
            175 * mm
        ],
    )

    authorization.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F0F9F6"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    GREEN,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
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
            ]
        )
    )

    story.append(
        authorization
    )

    # ============================================================
    # QR VERIFICATION
    # ============================================================

    story.append(
        Spacer(
            1,
            8,
        )
    )

    qr_buffer = create_qr(
        report_id
    )

    qr_image = RLImage(
        qr_buffer,
        width=25 * mm,
        height=25 * mm,
    )

    verification_table = Table(
        [
            [
                qr_image,
                Paragraph(
                    "<b>REPORT VERIFICATION</b><br/><br/>"
                    f"Report ID: {report_id}<br/>"
                    "This QR code identifies this "
                    "Medusa report.<br/><br/>"
                    "Status: FINAL / REVIEWED",
                    small_style,
                ),
            ]
        ],
        colWidths=[
            35 * mm,
            140 * mm,
        ],
    )

    verification_table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D7DCE1"),
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
            ]
        )
    )

    story.append(
        verification_table
    )

    # ============================================================
    # DISCLAIMER
    # ============================================================

    story.append(
        Spacer(
            1,
            8,
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor("#D7DCE1"),
        )
    )

    story.append(
        Spacer(
            1,
            5,
        )
    )

    story.append(
        Paragraph(
            "MEDUSA AI provides AI-assisted medical image screening "
            "and does not replace clinical judgment. The final "
            "interpretation contained in this report is based on "
            "radiologist review and should be considered together "
            "with the patient's clinical history and other relevant "
            "investigations.",
            small_style,
        )
    )

    # ============================================================
    # FOOTER
    # ============================================================

    def footer(canvas, doc):

        canvas.saveState()

        canvas.setStrokeColor(
            colors.HexColor("#D7DCE1")
        )

        canvas.line(
            15 * mm,
            12 * mm,
            195 * mm,
            12 * mm,
        )

        canvas.setFont(
            "Helvetica",
            7,
        )

        canvas.setFillColor(
            MID_GRAY
        )

        canvas.drawString(
            15 * mm,
            7 * mm,
            f"MEDUSA◉  |  Report ID: {report_id}",
        )

        canvas.drawRightString(
            195 * mm,
            7 * mm,
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
