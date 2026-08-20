import streamlit as st

from ui.background import set_background


def show_marketplace():

    set_background("marketplace.jpg")

    st.header("🛒 Marketplace")

    st.write(
        "Healthcare services and resources "
        "available through Medusa."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🩺 Healthcare Services"
        )

        st.write(
            "Find healthcare professionals, "
            "clinics and diagnostic services."
        )

        st.info(
            "Healthcare services coming soon."
        )

    with col2:

        st.subheader(
            "💊 Health Products"
        )

        st.write(
            "Access trusted health products "
            "and healthcare resources."
        )

        st.info(
            "Health marketplace coming soon."
        )

    st.divider()

    st.subheader("🔬 Diagnostics")

    st.write(
        "Medusa will connect users with "
        "diagnostic and laboratory services."
    )

    st.info(
        "Diagnostic marketplace coming soon."
    )
