import streamlit as st


def show_health():

    st.header("❤️ Health")

    st.write(
        "Your health information and "
        "AI analysis history."
    )


    # ========================================================
    # HEALTH OVERVIEW
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "AI Analyses",
            len(st.session_state.history),
        )

    with col2:

        st.metric(
            "Latest Result",
            (
                st.session_state.prediction
                if st.session_state.prediction
                else "None"
            ),
        )

    with col3:

        st.metric(
            "Status",
            "Active",
        )


    st.divider()


    # ========================================================
    # HISTORY
    # ========================================================

    st.subheader("Analysis History")

    history = st.session_state.history

    if not history:

        st.info(
            "No AI analysis has been completed yet."
        )

        return


    for item in reversed(history):

        if isinstance(item, dict):

            prediction = item.get(
                "prediction",
                "Unknown",
            )

            confidence = item.get(
                "confidence"
            )

            if confidence is not None:

                confidence_text = (
                    f"{confidence:.1%}"
                )

            else:

                confidence_text = "N/A"

            st.write(
                f"**{prediction}** "
                f"• Confidence: {confidence_text}"
            )

        else:

            st.write(str(item))
