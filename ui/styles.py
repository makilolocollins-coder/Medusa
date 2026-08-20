import streamlit as st


def load_styles():

    st.markdown(
        """
        <style>

        /* =========================
           GENERAL
        ========================= */

        .stApp {
            background: #F7F8FA;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }


        /* =========================
           HIDE STREAMLIT UI
        ========================= */

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }


        /* =========================
           BRAND
        ========================= */

        .brand {
            font-size: 30px;
            font-weight: 800;
            color: #101828;
            letter-spacing: -1px;
        }

        .brand span {
            color: #6D5DFB;
        }


        /* =========================
           STATUS
        ========================= */

        .status {
            display: inline-block;
            background: #ECFDF3;
            color: #027A48;
            padding: 7px 13px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
        }


        /* =========================
           HERO
        ========================= */

        .hero {
            padding: 45px 0 30px 0;
        }

        .hero h1 {
            font-size: 56px;
            line-height: 1.05;
            color: #101828;
            letter-spacing: -2px;
        }

        .hero span {
            color: #6D5DFB;
        }

        .hero p {
            max-width: 650px;
            color: #667085;
            font-size: 17px;
            line-height: 1.7;
        }


        /* =========================
           CARD
        ========================= */

        .card {
            background: white;
            border: 1px solid #EAECF0;
            border-radius: 22px;
            padding: 25px;
            box-shadow: 0 8px 30px rgba(16,24,40,0.04);
        }


        /* =========================
           AI CARD
        ========================= */

        .ai-card {
            background: linear-gradient(
                135deg,
                #12101F,
                #30265E
            );

            border-radius: 25px;
            padding: 35px;
            color: white;
        }

        .ai-label {
            font-size: 11px;
            letter-spacing: 2px;
            opacity: 0.7;
        }

        .ai-title {
            font-size: 32px;
            font-weight: 800;
            margin-top: 12px;
        }

        .ai-text {
            color: rgba(255,255,255,0.75);
            margin-top: 8px;
            line-height: 1.6;
        }


        /* =========================
           SECTION
        ========================= */

        .section {
            font-size: 25px;
            font-weight: 800;
            color: #101828;
            margin-top: 35px;
            margin-bottom: 18px;
        }


        /* =========================
           FOOTER
        ========================= */

        .medusa-footer {
            text-align: center;
            color: #98A2B3;
            font-size: 11px;
            margin-top: 60px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
