import streamlit as st
from PIL import Image

from ai.mammosense import (
    load_model,
    predict,
)


def show_detection():

    st.markdown(
        """
        <div class="hero">

            <h1>
                AI Detection
            </h1>

            <p>
                Upload a breast ultrasound image
                and let MammoSense analyse it.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    try:

        with st.spinner(
            "Loading MammoSense AI..."
        ):

            package = load_model()

        st.success(
            "MammoSense AI is ready."
        )

    except Exception as error:

        st.error(
            "MammoSense could not be loaded."
        )

        st.code(
            str(error)
        )

        return

    with st.expander(
        "Model information"
    ):

        st.write(
            f"**Model:** "
            f"{package['model_file']}"
        )

        st.write(
            f"**Architecture:** "
            f"{package['architecture']}"
        )

        st.write(
            f"**Classes:** "
            f"{', '.join(package['classes'])}"
        )

        st.write(
            f"**Device:** "
            f"{package['device']}"
        )

    uploaded = st.file_uploader(
        "Upload breast ultrasound image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded is None:
        return

    image = Image.open(
        uploaded
    )

    left, right = st.columns(
        [1.2, 1]
    )

    with left:

        st.image(
            image,
            caption="Ultrasound image",
            use_container_width=True,
        )

    with right:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Image ready
                </div>

                <br>

                <div class="card-text">
                    Your image is ready for
                    MammoSense analysis.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Analyse with MammoSense →",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Medusa is analysing..."
            ):

                result = predict(
                    image
                )

            st.session_state.prediction = result

            st.session_state.history.append(
                {
                    "prediction":
                        result["prediction"],

                    "confidence":
                        result["confidence"],
                }
            )

            st.rerun()

    result = (
        st.session_state.prediction
    )

    if result is None:
        return

    st.markdown("---")

    st.markdown(
        '<div class="section">'
        'Analysis Result'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="result">

            <div class="result-small">
                MammoSense classification
            </div>

            <div class="result-name">
                {result["prediction"]}
            </div>

            <div class="result-confidence">
                Confidence:
                {result["confidence"] * 100:.2f}%
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section">'
        'Probability Distribution'
        '</div>',
        unsafe_allow_html=True,
    )

    for (
        name,
        probability,
    ) in result[
        "probabilities"
    ].items():

        st.write(
            f"**{name}**"
        )

        st.progress(
            probability
        )

        st.caption(
            f"{probability * 100:.2f}%"
        )

    st.markdown(
        """
        <div class="warning">

        ⚠️ <b>Medical notice:</b>
        Medusa provides AI-assisted screening
        support. This result is not a diagnosis
        and should be reviewed by a qualified
        healthcare professional.

        </div>
        """,
        unsafe_allow_html=True,
    )
