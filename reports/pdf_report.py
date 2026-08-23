import streamlit as st
from utils.supabase_client import get_supabase


SCAN_BUCKET = "mammosense-scans"
REPORT_BUCKET = "medical-reports"


def get_user():
    try:
        response = get_supabase().auth.get_user()
        return response.user if response else None
    except Exception:
        return None


def show_pdf_reports():

    st.title("Medical Reports")
    st.caption("Radiologist-approved medical reports")

    supabase = get_supabase()
    user = get_user()

    if not user:
        st.error("Please log in to view your reports.")
        return

    # ============================================================
    # FIND PATIENT REPORTS
    # ============================================================

    try:
        response = (
            supabase
            .table("medical_reports")
            .select(
                "id,report_id,scan_id,review_id,user_id,"
                "patient_id,patient_name,pdf_path,status,"
                "approved_at,created_at"
            )
            .eq("status", "APPROVED")
            .order("approved_at", desc=True)
            .execute()
        )

        all_reports = response.data or []

    except Exception as e:
        st.error("Unable to load medical reports.")
        st.exception(e)
        return

    # ------------------------------------------------------------
    # Match reports to the logged-in patient.
    #
    # user_id is checked first, but patient_id is also supported.
    # This prevents reports from disappearing if the report was
    # created by the radiologist account.
    # ------------------------------------------------------------

    reports = [
        r for r in all_reports
        if r.get("user_id") == user.id
        or r.get("patient_id")
    ]

    # If patient_id is not an auth UUID, use the patient's scans
    # belonging to this logged-in account to identify ownership.

    if reports:

        try:
            scan_response = (
                supabase
                .table("ai_scans")
                .select("id,patient_id")
                .eq("user_id", user.id)
                .execute()
            )

            user_scans = scan_response.data or []
            user_scan_ids = {
                x.get("id") for x in user_scans
            }
            user_patient_ids = {
                x.get("patient_id") for x in user_scans
            }

            reports = [
                r for r in reports
                if (
                    r.get("user_id") == user.id
                    or r.get("scan_id") in user_scan_ids
                    or r.get("patient_id") in user_patient_ids
                )
            ]

        except Exception:
            reports = [
                r for r in reports
                if r.get("user_id") == user.id
            ]

    if not reports:
        st.info(
            "No radiologist-approved medical reports "
            "are available yet."
        )
        return

    st.success(
        f"{len(reports)} approved report"
        + ("s" if len(reports) != 1 else "")
        + " available."
    )

    # ============================================================
    # REPORTS
    # ============================================================

    for report in reports:

        report_id = (
            report.get("report_id")
            or report.get("id")
            or "N/A"
        )

        scan_id = report.get("scan_id")
        patient_id = report.get("patient_id") or "N/A"
        patient_name = (
            report.get("patient_name")
            or "N/A"
        )

        pdf_path = report.get("pdf_path")
        approved_at = (
            report.get("approved_at")
            or "N/A"
        )

        # --------------------------------------------------------
        # GET SCAN
        # --------------------------------------------------------

        scan = {}

        if scan_id:

            try:

                result = (
                    supabase
                    .table("ai_scans")
                    .select(
                        "id,user_id,patient_id,patient_name,"
                        "patient_state,examination,model,"
                        "prediction,confidence,probabilities,"
                        "image_path,status,created_at"
                    )
                    .eq("id", scan_id)
                    .limit(1)
                    .execute()
                )

                if result.data:
                    scan = result.data[0]

            except Exception:
                pass

        # Use scan information as fallback.
        patient_name = (
            patient_name
            if patient_name != "N/A"
            else scan.get("patient_name", "N/A")
        )

        patient_id = (
            patient_id
            if patient_id != "N/A"
            else scan.get("patient_id", "N/A")
        )

        # ========================================================
        # REPORT CARD
        # ========================================================

        with st.container(border=True):

            st.subheader(
                f"MEDUSA AI REPORT"
            )

            st.success("RADIOLOGIST APPROVED")

            # ----------------------------------------------------
            # PATIENT INFORMATION
            # ----------------------------------------------------

            st.markdown("### Patient Information")

            c1, c2 = st.columns(2)

            with c1:
                st.write(
                    f"**Patient Name:** {patient_name}"
                )

                st.write(
                    f"**Patient ID:** {patient_id}"
                )

                st.write(
                    f"**State:** "
                    f"{scan.get('patient_state', 'N/A')}"
                )

            with c2:

                st.write(
                    f"**Report ID:** {report_id}"
                )

                st.write(
                    f"**Review ID:** "
                    f"{report.get('review_id') or 'N/A'}"
                )

                st.write(
                    f"**Approved:** {approved_at}"
                )

            # ----------------------------------------------------
            # EXAMINATION
            # ----------------------------------------------------

            st.divider()
            st.markdown("### Examination")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.write(
                    f"**Examination:** "
                    f"{scan.get('examination', 'N/A')}"
                )

            with c2:
                st.write(
                    f"**Model:** "
                    f"{scan.get('model', 'N/A')}"
                )

            with c3:

                confidence = scan.get("confidence")

                if confidence is not None:

                    try:
                        st.write(
                            f"**AI Confidence:** "
                            f"{float(confidence):.1%}"
                        )
                    except Exception:
                        st.write(
                            f"**AI Confidence:** "
                            f"{confidence}"
                        )

                else:
                    st.write(
                        "**AI Confidence:** N/A"
                    )

            prediction = scan.get("prediction")

            if prediction:
                st.info(
                    f"AI Finding: {prediction}"
                )

            # ----------------------------------------------------
            # ORIGINAL SCAN
            # ----------------------------------------------------

            image_path = scan.get("image_path")

            if image_path:

                st.markdown(
                    "### Examination Image"
                )

                try:

                    image_bytes = (
                        supabase
                        .storage
                        .from_(SCAN_BUCKET)
                        .download(image_path)
                    )

                    if image_bytes:

                        st.image(
                            image_bytes,
                            caption="Original Examination Scan",
                            use_container_width=True,
                        )

                    else:

                        st.warning(
                            "The examination image could not "
                            "be retrieved."
                        )

                except Exception as e:

                    st.warning(
                        "The examination image could not "
                        "be loaded."
                    )

            # ----------------------------------------------------
            # PDF DOWNLOAD
            # ----------------------------------------------------

            st.divider()
            st.markdown(
                "### Final Medical Report"
            )

            if not pdf_path:

                st.warning(
                    "This report has been approved, but "
                    "the PDF has not been attached yet."
                )

                continue

            try:

                pdf_bytes = (
                    supabase
                    .storage
                    .from_(REPORT_BUCKET)
                    .download(pdf_path)
                )

                if not pdf_bytes:

                    st.warning(
                        "The approved PDF could not be retrieved."
                    )

                    continue

                st.download_button(
                    label="Download Final Medical Report",
                    data=pdf_bytes,
                    file_name=f"{report_id}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key=f"download_{report.get('id')}",
                )

                st.caption(
                    "This report was approved by a qualified "
                    "radiologist."
                )

            except Exception as e:

                st.error(
                    "The report is approved, but the PDF "
                    "could not be downloaded."
                )

                st.exception(e)
