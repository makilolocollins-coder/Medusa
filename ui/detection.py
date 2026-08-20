import streamlit as st
from PIL import Image

from ai.mammosense import get_mammosense


def show_detection():

    # ========================================================
    # HEADER
    # ========================================================

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

    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        with st.spinner("Loading MammoSense AI..."):

            engine = get_mammosense()

        st.success("MammoSense AI is ready.")

    except Exception as error:

        st.error("MammoSense could not be loaded.")
        st.code(str(error))

        return

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander("Model information"):

        st.write(
            "**Model:** mammosense_v2.pt"
        )

        st.write(
            f"**Architecture:** "
            f"{engine.architecture}"
        )

        st.write(
            f"**Classes:** "
            f"{', '.join(engine.classes)}"
        )

        st.write(
            f"**Image size:** "
            f"{engine.image_size} × "
            f"{engine.image_size}"
        )

        st.write(
            f"**Device:** "
            f"{engine.device}"
        )

    # ========================================================
    # UPLOAD
    # ========================================================

    st.markdown(
"""
<div class="section">
    Upload Ultrasound
</div>
""",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose a breast ultrasound image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded is None:

        st.info(
            "Upload an ultrasound image to begin analysis."
        )

        return

    # ========================================================
    # READ IMAGE
    # ========================================================

    try:

        image = Image.open(
            uploaded
        ).convert("RGB")

    except Exception as error:

        st.error(
            "The uploaded file could not be read."
        )

        st.code(
            str(error)
        )

        return

    # ========================================================
    # IMAGE
    # ========================================================

    left, right = st.columns(
        [1.15, 1]
    )

    with left:

        st.markdown(
"""
<div class="card">

    <div class="card-title">
        Ultrasound Image
    </div>

    <div class="card-text">
        Image uploaded for AI analysis.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True,
        )

    # ========================================================
    # ANALYSIS
    # ========================================================

    with right:

        st.markdown(
"""
<div class="card">

    <div class="card-title">
        MammoSense Analysis
    </div>

    <div class="card-text">
        The image will be processed using
        the MammoSense V2 model.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

        analyse = st.button(
            "Analyse Image →",
            type="primary",
            use_container_width=True,
        )

        if not analyse:
            return

        try:

            with st.spinner(
                "MammoSense is analysing the image..."
            ):

                result = engine.predict(
                    image
                )

        except Exception as error:

            st.error(
                "MammoSense analysis failed."
            )

            st.code(
                str(error)
            )

            return

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
"""
<div class="section">
    Analysis Result
</div>
""",
        unsafe_allow_html=True,
    )

    prediction = result["prediction"]

    confidence = result["confidence"]

    probabilities = result["probabilities"]

    st.markdown(
f"""
<div class="result">

    <div class="result-small">
        MammoSense Prediction
    </div>

    <div class="result-name">
        {prediction}
    </div>

    <div class="result-confidence">
        Confidence: {confidence * 100:.2f}%
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.markdown(
"""
<div class="section">
    Class Probabilities
</div>
""",
        unsafe_allow_html=True,
    )

    columns = st.columns(
        len(probabilities)
    )

    for column, (
        class_name,
        probability,
    ) in zip(
        columns,
        probabilities.items(),
    ):

        with column:

            st.metric(
                class_name,
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
"""
<div class="warning">

    <strong>Important:</strong>
    MammoSense is an AI research and
    decision-support tool. Its prediction
    is not a medical diagnosis and should
    not replace assessment by a qualified
    healthcare professional.

</div>
""",
        unsafe_allow_html=True,
    )
