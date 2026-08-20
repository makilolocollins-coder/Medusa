import streamlit as st

from ui.background import set_background


def show_health():

    set_background("health.jpg")

    st.header("❤️ Health")

    st.write("Health page is running the NEW version.")

    history = st.session_state.get("history", [])
    prediction = st.session_state.get("prediction", None)

    st.write("History records:", len(history))
    st.write("Latest prediction:", prediction)

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
                "confidence",
                None,
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
