import streamlit as st

from ui.background import set_background


def show_health():

    set_background("health.jpg")

    st.header("❤️ Health")

    st.write(
        "Your health information and "
        "AI analysis history."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "AI Analyses",
            len(st.session_state.history),
        )

    with col2:

        latest = (
            st.session_state.prediction
            if st.session_state.prediction
            else "None"
        )

        st.metric(
            "Latest Result",
            latest,
        )

    with col3:

        st.metric(
            "Status",
            "Active",
        )

    st.divider()

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
                f"• Confidence: "
                f"{confidence_text}"
            )

        else:

            st.write(str(item))
