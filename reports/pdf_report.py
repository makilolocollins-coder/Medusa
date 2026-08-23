import streamlit as st
from utils.supabase_client import get_supabase


def show_pdf_reports():

    st.title("Medical Reports")
    st.caption("Radiologist-approved medical reports")

    # ------------------------------------------------------------
    # LOGIN
    # ------------------------------------------------------------

    try:
        supabase = get_supabase()
        auth = supabase.auth.get_user()
        user = auth.user if auth else None
    except Exception:
        user = None

    if not user:
        st.error("Please log in to view your reports.")
        return

    # ------------------------------------------------------------
    # GET APPROVED REPORTS
    # ------------------------------------------------------------

    try:
        response = (
            supabase
            .table("medical_reports")
            .select(
                "id,report_id,scan_id,review_id,user_id,"
                "patient_id,patient_name,pdf_path,status,"
                "approved_at,created_at"
            )
            .eq("user_id", user.id)
            .eq("status", "APPROVED")
            .order("approved_at", desc=True)
            .execute()
        )

        reports = response.data or []

    except Exception as e:
        st.error("Unable to load medical reports.")
        st.exception(e)
        return

    if not reports:
        st.info(
            "No radiologist-approved medical reports are "
            "available yet."
        )
        return

    st.success(
        f"{len(reports)} approved report"
        + ("s" if len(reports) != 1 else "")
        + " available."
    )

    # ------------------------------------------------------------
    # DISPLAY REPORTS
    # ------------------------------------------------------------

    for report in reports:

        report_id = report.get("report_id") or report.get("id")
        scan_id = report.get("scan_id")
        patient_id = report.get("patient_id")
        patient_name = report.get("patient_name") or "Unknown"
        pdf_path = report.get("pdf_path")
        approved_at = report.get("approved_at")

        # --------------------------------------------------------
        # GET SCAN
        # --------------------------------------------------------

        scan = {}

        if scan_id:

            try:
                scan_response = (
                    supabase
                    .table("ai_scans")
                    .select(
                        "id,patient_id,patient_name,patient_state,"
                        "examination,model,prediction,confidence,"
                        "probabilities,image_path,status,created_at"
                    )
                    .eq("id", scan_id)
                    .limit(1)
                    .execute()
                )

                scan_data = scan_response.data or []

                if scan_data:
                    scan = scan_data[0]

            except Exception:
                scan = {}

        # --------------------------------------------------------
        # REPORT CARD
        # --------------------------------------------------------

        with st.container(border=True):

            st.subheader(
                f"Report {report_id}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    f"**Patient:** "
                    f"{patient_name}"
                )

                st.markdown(
                    f"**Patient ID:** "
                    f"{patient_id or scan.get('patient_id', 'N/A')}"
                )

                st.markdown(
                    f"**Examination:** "
                    f"{scan.get('examination', 'N/A')}"
                )

            with col2:

                st.markdown(
                    f"**Report ID:** "
                    f"{report_id}"
                )

                st.markdown(
                    f"**Approved:** "
                    f"{approved_at or 'N/A'}"
                )

                st.markdown(
                    f"**Review ID:** "
                    f"{report.get('review_id') or 'N/A'}"
                )

            # ----------------------------------------------------
            # SCAN INFORMATION
            # ----------------------------------------------------

            if scan:

                st.divider()

                st.markdown("### Examination")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.write(
                        f"**AI Finding:** "
                        f"{scan.get('prediction', 'N/A')}"
                    )

                with c2:
                    confidence = scan.get("confidence")

                    if confidence is not None:
                        try:
                            confidence = float(confidence)
                            st.write(
                                f"**AI Confidence:** "
                                f"{confidence:.1%}"
                            )
                        except Exception:
                            st.write(
                                f"**AI Confidence:** "
                                f"{confidence}"
                            )

                with c3:
                    st.write(
                        f"**Model:** "
                        f"{scan.get('model', 'N/A')}"
                    )

                # ------------------------------------------------
                # SHOW ORIGINAL SCAN
                # ------------------------------------------------

                image_path = scan.get("image_path")

                if image_path:

                    try:

                        image_bytes = (
                            supabase
                            .storage
                            .from_("mammosense-scans")
                            .download(image_path)
                        )

                        if image_bytes:

                            st.image(
                                image_bytes,
                                caption="Examination Image",
                                use_container_width=True,
                            )

                    except Exception:
                        st.warning(
                            "The examination image could not "
                            "be loaded."
                        )

            # ----------------------------------------------------
            # PDF
            # ----------------------------------------------------

            st.divider()

            if not pdf_path:

                st.warning(
                    "This report is approved, but its PDF "
                    "file has not been stored yet."
                )

                continue

            try:

                pdf_bytes = (
                    supabase
                    .storage
                    .from_("medical-reports")
                    .download(pdf_path)
                )

                if not pdf_bytes:

                    st.warning(
                        "The approved report PDF is empty."
                    )
                    continue

                st.download_button(
                    "Download Final Medical Report",
                    data=pdf_bytes,
                    file_name=f"{report_id}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key=f"download_report_{report.get('id')}",
                )

                st.success(
                    "Radiologist-approved report available."
                )

            except Exception as e:

                st.error(
                    "The report exists, but the PDF could "
                    "not be downloaded."
                )

                st.exception(e)
