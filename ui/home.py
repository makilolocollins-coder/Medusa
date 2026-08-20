import streamlit as st

from ui.background import set_background


def show_home():

    set_background("home.jpg")

    st.header(
        "Your health, intelligently connected."
    )

    st.write(
        "Medusa combines artificial intelligence, "
        "health insights and healthcare services "
        "in one intelligent platform."
    )

    st.write("")

    st.subheader(
        "🧠 Medusa Intelligence"
    )

    st.info(
        "MammoSense\n\n"
        "AI-assisted breast ultrasound analysis."
    )

    if st.button(
        "Start AI Analysis →",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.page = "AI Detection"

        st.rerun()

    st.subheader("AI Models")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🧬 MammoSense")

        st.write(
            "Breast ultrasound AI "
            "classification."
        )

        st.success("Available")

    with col2:

        st.markdown("### 🧠 Prostate AI")

        st.write(
            "Multimodal prostate MRI "
            "intelligence."
        )

        st.info("Coming soon")

    with col3:

        st.markdown("### ✦ More AI")

        st.write(
            "Additional medical AI "
            "systems."
        )

        st.info("Coming soon")

    st.divider()

    st.caption(
        "AI-assisted screening only. "
        "Not a substitute for professional medical advice."
    )
