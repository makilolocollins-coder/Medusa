import streamlit as st


def load_styles():

    st.markdown(
        """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #F7F8FA;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* BRAND */

.brand {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1.3px;
    color: #101828;
}

.brand span {
    color: #6D5DFB;
}


/* STATUS */

.status {
    display: inline-block;
    background: #ECFDF3;
    color: #027A48;
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
}


/* HERO */

.hero {
    padding: 45px 0 30px 0;
}

.hero h1 {
    font-size: clamp(38px, 6vw, 64px);
    line-height: 1.02;
    letter-spacing: -3px;
    color: #101828;
}

.hero h1 span {
    color: #6D5DFB;
}

.hero p {
    max-width: 680px;
    color: #667085;
    font-size: 17px;
    line-height: 1.7;
}


/* CARDS */

.card {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 24px;
    padding: 26px;
    box-shadow:
        0 8px 30px rgba(16,24,40,.035);
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #101828;
}

.card-text {
    color: #667085;
    font-size: 13px;
    line-height: 1.6;
}


/* AI CARD */

.ai-card {
    background:
        linear-gradient(
            135deg,
            #12101F,
            #30265E
        );

    border-radius: 28px;
    padding: 38px;
    color: white;
    min-height: 250px;
}

.ai-label {
    font-size: 11px;
    letter-spacing: 1.7px;
    opacity: .65;
}

.ai-title {
    font-size: 34px;
    font-weight: 800;
    margin-top: 13px;
}

.ai-text {
    max-width: 580px;
    color: rgba(255,255,255,.72);
    line-height: 1.7;
    font-size: 14px;
}


/* MODELS */

.model-card {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 22px;
    padding: 24px;
    min-height: 170px;
}

.model-icon {
    font-size: 30px;
}

.model-name {
    font-weight: 700;
    margin-top: 12px;
    color: #101828;
}

.model-description {
    color: #667085;
    font-size: 13px;
    margin-top: 7px;
}


/* RESULT */

.result {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 28px;
    padding: 35px;
    text-align: center;
}

.result-small {
    color: #667085;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.result-name {
    color: #101828;
    font-size: 44px;
    font-weight: 800;
    margin-top: 8px;
}

.result-confidence {
    color: #667085;
}


/* MARKETPLACE */

.market {
    background: white;
    border: 1px solid #EAECF0;
    border-radius: 22px;
    padding: 25px;
    min-height: 180px;
}

.market-icon {
    font-size: 30px;
}

.market-title {
    font-size: 17px;
    font-weight: 700;
    margin-top: 12px;
}

.market-text {
    color: #667085;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 7px;
}


/* WARNING */

.warning {
    background: #FFF8E7;
    border: 1px solid #F2CC72;
    color: #694B00;
    border-radius: 15px;
    padding: 15px;
    font-size: 12px;
    line-height: 1.6;
}


/* SECTION */

.section {
    font-size: 25px;
    font-weight: 800;
    color: #101828;
    letter-spacing: -.8px;
    margin-top: 38px;
    margin-bottom: 18px;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #98A2B3;
    font-size: 11px;
    margin-top: 70px;
}

</style>
""",
        unsafe_allow_html=True,
    )
