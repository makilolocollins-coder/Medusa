import streamlit as st


def show_detection():

    st.title("AI Detection")

    st.markdown(
        "<h1>HTML TEST</h1>",
        unsafe_allow_html=True,
    )

    st.write(
        "If you can see the words HTML TEST as a large heading, "
        "HTML rendering is working."
    )
