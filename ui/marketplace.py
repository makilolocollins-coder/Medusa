import streamlit as st


def show_marketplace():

    st.header("🛒 Marketplace")

    st.write(
        "Healthcare services, products and "
        "resources available through Medusa."
    )


    # ========================================================
    # SERVICES
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.subheader("🩺 Healthcare")

        st.write(
            "Connect with healthcare "
            "professionals and services."
        )

        st.button(
            "Explore",
            key="healthcare",
        )


    with col2:

        st.subheader("💊 Pharmacy")

        st.write(
            "Access trusted healthcare "
            "products and resources."
        )

        st.button(
            "Explore",
            key="pharmacy",
        )


    with col3:

        st.subheader("🔬 Diagnostics")

        st.write(
            "Explore diagnostic and "
            "laboratory services."
        )

        st.button(
            "Explore",
            key="diagnostics",
        )


    st.divider()


    # ========================================================
    # COMING SOON
    # ========================================================

    st.info(
        "Marketplace services are being "
        "prepared for Medusa."
    )
