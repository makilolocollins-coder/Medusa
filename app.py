    import streamlit as st

from ui.styles import load_styles


st.set_page_config(
    page_title="Medusa AI",
    page_icon="🧬",
    layout="wide",
)

load_styles()


# ============================================================
# HEADER
# ============================================================

col1, col2 = st.columns([4, 1])

with col1:
    st.title("MEDUSA AI")

with col2:
    st.success("AI ONLINE")


# ============================================================
# HERO
# ============================================================

st.header(
    "Your health, intelligently connected."
)

st.write(
    "Medusa combines artificial intelligence, "
    "health insights and healthcare services "
    "in one intelligent platform."
)


# ============================================================
# AI CARD
# ============================================================

st.subheader("MEDUSA INTELLIGENCE")

st.info(
    "AI Health Detection\n\n"
    "MammoSense breast ultrasound intelligence "
    "is coming next."
)


# ============================================================
# TEST
# ============================================================

st.success(
    "Step 2: UI is working."
)
