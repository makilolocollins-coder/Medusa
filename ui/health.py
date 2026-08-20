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

    history = st.session_state.history
    prediction = st.session_state.prediction

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "AI Analyses",
            len(history),
        )

    with col2:
        st.metric(
            "Latest Result",
            prediction if prediction else "None",
        )

    with col3:
        st.metric(
            "Status",
            "Active",
        )

    st.divider()

    st.subheader("Analysis History")

    if not history:
        st.info(
            "No AI analysis has been completed yet."
        )
        return

    for item in reversed(history):

        if isinstance(item, dict):

            result = item.get(
                "prediction",
                "Unknown",
            )

            confidence = item.get(
                "confidence"
            )

            if confidence is not None:
                confidence_text = f"{confidence:.1%}"
            else:
                confidence_text = "N/A"

            st.write(
                f"**{result}** • "
                f"Confidence: {confidence_text}"
            )

        else:
            st.write(str(item))
