import streamlit as st


def show_home():

    # ========================================================
    # HERO
    # ========================================================

    st.markdown(
"""
<div class="hero">

    <h1>
        Your health,<br>
        <span>intelligently connected.</span>
    </h1>

    <p>
        Medusa combines artificial intelligence,
        health insights and healthcare services
        in one intelligent platform.
    </p>

</div>
""",
        unsafe_allow_html=True,
    )

    # ========================================================
    # AI CARD
    # ========================================================

    st.markdown(
"""
<div class="ai-card">

    <div class="ai-label">
        MEDUSA INTELLIGENCE
    </div>

    <div class="ai-title">
        AI Health Detection
    </div>

    <div class="ai-text">
        Start with MammoSense, Medusa's
        breast ultrasound intelligence engine.
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ========================================================
    # START AI
    # ========================================================

    if st.button(
        "Start AI Analysis →",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.page = "AI Detection"

        st.rerun()

    # ========================================================
    # MODELS
    # ========================================================

    st.markdown(
"""
<div class="section">
    AI Models
</div>
""",
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    # ========================================================
    # MAMMOSENSE
    # ========================================================

    with a:

        st.markdown(
"""
<div class="model-card">

    <div class="model-icon">
        🧬
    </div>

    <div class="model-name">
        MammoSense
    </div>

    <div class="model-description">
        Breast ultrasound AI
        classification.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

    # ========================================================
    # PROSTATE
    # ========================================================

    with b:

        st.markdown(
"""
<div class="model-card">

    <div class="model-icon">
        🧠
    </div>

    <div class="model-name">
        Prostate AI
    </div>

    <div class="model-description">
        Multimodal prostate MRI
        intelligence.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

    # ========================================================
    # FUTURE MODELS
    # ========================================================

    with c:

        st.markdown(
"""
<div class="model-card">

    <div class="model-icon">
        ✦
    </div>

    <div class="model-name">
        Medusa Intelligence
    </div>

    <div class="model-description">
        More medical AI models
        coming soon.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )
