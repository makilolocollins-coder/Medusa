import streamlit as st

from ui.styles import load_styles


st.set_page_config(
    page_title="Medusa AI",
    page_icon="🧬",
    layout="wide",
)


load_styles()


# =========================
# HEADER
# =========================

left, right = st.columns([4, 1])

with left:

    st.markdown(
        """
        <div class="brand">
            MEDUSA<span>◉</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:

    st.markdown(
        """
        <div style="text-align:right;">
            <span class="status">
                ● AI ONLINE
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# HERO
# =========================

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


# =========================
# TEST CARD
# =========================

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
            MammoSense breast ultrasound
            intelligence is coming next.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="medusa-footer">
        MEDUSA AI<br>
        Intelligent Health Infrastructure
    </div>
    """,
    unsafe_allow_html=True,
)
