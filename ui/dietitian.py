# ============================================================
# MEDUSA AI
# DIETITIAN / NUTRITION AI
#
# AI ENGINE:
# DeepSeek API
#
# IMPORTANT:
# - Never put the API key directly in this file.
# - Configure DEEPSEEK_API_KEY in Streamlit Secrets.
# - The module also works in offline/demo mode without an API key.
#
# ============================================================

import os
import re
from datetime import datetime

import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Medusa Nutrition AI"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Current DeepSeek API model.
# Can be overridden through Streamlit secrets/environment.
DEFAULT_MODEL = "deepseek-v4-flash"

MAX_HISTORY_MESSAGES = 20


# ============================================================
# SAFE NUTRITION SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Medusa Nutrition AI, an AI nutrition assistant integrated
into the Medusa health platform.

Your job is to provide useful, understandable and practical
nutrition information.

You can help with:

- General nutrition questions
- Healthy eating
- Meal planning
- Nigerian and African foods
- Food substitutions
- Protein
- Fibre
- Vitamins and minerals
- Hydration
- Weight management
- Sports nutrition
- Food labels
- Portion guidance
- Meal ideas
- Budget-friendly meals
- Nutrition education
- Nutrition questions related to common health conditions
- Preparing questions for a qualified dietitian

IMPORTANT MEDICAL SAFETY RULES:

1. You are NOT a doctor or registered dietitian.

2. Do not claim to diagnose a disease.

3. Do not tell a user to stop prescribed medication.

4. Do not prescribe medication.

5. Do not provide dangerous crash diets.

6. Do not encourage starvation, purging or extreme calorie
   restriction.

7. For pregnancy, infants, children, eating disorders, severe
   kidney disease, severe liver disease, insulin-dependent
   diabetes, cancer treatment, severe malnutrition, or other
   high-risk situations, provide general educational information
   and recommend professional clinical assessment.

8. If the user describes a medical emergency, advise them to seek
   urgent medical care rather than trying to manage the emergency
   through nutrition advice.

9. If a Medusa AI examination result is supplied, treat it as
   screening information, NOT a definitive diagnosis.

10. Never tell the patient that an AI result proves that they have
    a disease.

11. When discussing a disease, explain that nutrition is supportive
    and does not replace appropriate medical treatment.

12. Avoid pretending to know the patient's laboratory values,
    medications, allergies, weight or medical history unless those
    details were explicitly provided.

13. Ask sensible follow-up questions when information is missing.

STYLE:

- Be warm and professional.
- Use simple language.
- Give practical examples.
- Prefer locally available foods when appropriate.
- Consider Nigerian foods when the user is in Nigeria.
- Use headings and bullet points when useful.
- Do not unnecessarily overwhelm the patient.
- Clearly distinguish facts from estimates.
"""


# ============================================================
# OPTIONAL DEEPSEEK IMPORT
# ============================================================

try:

    from openai import OpenAI

except ImportError:

    OpenAI = None


# ============================================================
# API KEY
# ============================================================

def get_deepseek_api_key():
    """
    Read the DeepSeek API key securely.

    Priority:
    1. Streamlit secrets
    2. Environment variable
    """

    try:

        key = st.secrets.get(
            "DEEPSEEK_API_KEY",
            None,
        )

        if key:

            return str(key).strip()

    except Exception:

        pass

    key = os.getenv(
        "DEEPSEEK_API_KEY"
    )

    if key:

        return str(key).strip()

    return ""


# ============================================================
# MODEL
# ============================================================

def get_model_name():

    try:

        model = st.secrets.get(
            "DEEPSEEK_MODEL",
            None,
        )

        if model:

            return str(model).strip()

    except Exception:

        pass

    return os.getenv(
        "DEEPSEEK_MODEL",
        DEFAULT_MODEL,
    )


# ============================================================
# API STATUS
# ============================================================

def deepseek_available():

    if OpenAI is None:

        return False

    return bool(
        get_deepseek_api_key()
    )


# ============================================================
# DEEPSEEK CLIENT
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_deepseek_client():

    api_key = get_deepseek_api_key()

    if not api_key:

        return None

    if OpenAI is None:

        return None

    return OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():

    defaults = {

        "nutrition_messages": [],

        "nutrition_profile": {},

        "nutrition_score": None,

        "nutrition_assessment": None,

        "nutrition_meal_plan": None,

        "dietitian_review_requested": False,

        "nutrition_question": "",

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


# ============================================================
# PATIENT PROFILE
# ============================================================

def show_nutrition_profile():

    st.subheader(
        "Patient Nutrition Profile"
    )

    st.caption(
        "Optional information that helps Medusa personalize "
        "nutrition guidance."
    )

    existing = st.session_state.get(
        "nutrition_profile",
        {},
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=int(
                existing.get(
                    "age",
                    25,
                )
            ),
        )

        sex = st.selectbox(
            "Sex",
            [
                "Prefer not to say",
                "Male",
                "Female",
            ],
            index=[
                "Prefer not to say",
                "Male",
                "Female",
            ].index(
                existing.get(
                    "sex",
                    "Prefer not to say",
                )
            )
            if existing.get(
                "sex",
                "Prefer not to say",
            )
            in [
                "Prefer not to say",
                "Male",
                "Female",
            ]
            else 0,
        )

        activity = st.selectbox(
            "Physical activity",
            [
                "Low",
                "Moderate",
                "High",
            ],
        )

    with col2:

        goal = st.selectbox(
            "Main nutrition goal",
            [
                "General healthy eating",
                "Weight management",
                "Muscle gain/support",
                "Better energy",
                "Sports performance",
                "Nutrition during illness",
                "Other",
            ],
        )

        dietary_pattern = st.selectbox(
            "Dietary pattern",
            [
                "No specific restriction",
                "Vegetarian",
                "Vegan",
                "Low-meat",
                "Other",
            ],
        )

        allergies = st.text_input(
            "Food allergies/intolerances",
            value=existing.get(
                "allergies",
                "",
            ),
            placeholder="e.g. peanuts, milk",
        )

    medical_conditions = st.text_area(
        "Medical conditions relevant to nutrition",
        value=existing.get(
            "medical_conditions",
            "",
        ),
        placeholder=(
            "Optional. Example: hypertension, diabetes, "
            "high cholesterol."
        ),
    )

    if st.button(
        "Save Nutrition Profile",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.nutrition_profile = {

            "age": age,

            "sex": sex,

            "activity": activity,

            "goal": goal,

            "dietary_pattern": dietary_pattern,

            "allergies": allergies,

            "medical_conditions": medical_conditions,
        }

        st.success(
            "Nutrition profile saved."
        )


# ============================================================
# BUILD PATIENT CONTEXT
# ============================================================

def build_patient_context():

    profile = st.session_state.get(
        "nutrition_profile",
        {},
    )

    context = ""

    if profile:

        context += (
            "\nPATIENT NUTRITION PROFILE:\n"
        )

        for key, value in profile.items():

            if value:

                context += (
                    f"- {key}: {value}\n"
                )

    # --------------------------------------------------------
    # Existing Medusa scan
    # --------------------------------------------------------

    scan_result = st.session_state.get(
        "scan_result"
    )

    if scan_result:

        context += (
            "\nMEDUSA AI SCREENING CONTEXT:\n"
        )

        if isinstance(
            scan_result,
            dict,
        ):

            safe_fields = [
                "prediction",
                "predicted_class",
                "confidence",
                "model_name",
                "architecture",
            ]

            for field in safe_fields:

                if field in scan_result:

                    context += (
                        f"- {field}: "
                        f"{scan_result[field]}\n"
                    )

        context += (
            "IMPORTANT: This is AI-assisted screening "
            "information and is not a definitive diagnosis.\n"
        )

    return context


# ============================================================
# OFFLINE RESPONSE
# ============================================================

def offline_response(
    question,
):

    q = question.lower().strip()

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "water",
            "hydration",
            "drink",
        ]
    ):

        return (
            "Hydration needs vary with body size, activity, "
            "temperature and health status. A practical approach "
            "is to drink regularly through the day and increase "
            "fluids when you are sweating or exercising. If you "
            "have a condition that requires fluid restriction, "
            "follow your clinician's advice."
        )

    # --------------------------------------------------------
    # Protein
    # --------------------------------------------------------

    if "protein" in q:

        return (
            "Good protein sources include beans, lentils, eggs, "
            "fish, chicken, soy foods, milk and yoghurt. For a "
            "balanced meal, combine a protein source with "
            "vegetables or fruit and an appropriate staple such "
            "as rice, yam, potatoes or whole grains."
        )

    # --------------------------------------------------------
    # Weight loss
    # --------------------------------------------------------

    if (
        "lose weight" in q
        or "weight loss" in q
        or "slim" in q
    ):

        return (
            "For sustainable weight management, focus on regular "
            "balanced meals, appropriate portions, vegetables, "
            "protein, whole or minimally processed foods and "
            "regular physical activity. Avoid crash diets and "
            "extreme calorie restriction."
        )

    # --------------------------------------------------------
    # Nigerian foods
    # --------------------------------------------------------

    if any(
        food in q
        for food in [
            "jollof",
            "eba",
            "garri",
            "fufu",
            "yam",
            "beans",
            "plantain",
        ]
    ):

        return (
            "Nigerian foods can absolutely fit into a healthy "
            "diet. Portion size and the overall meal matter. "
            "For example, pair a moderate portion of a staple "
            "such as rice, yam, eba or plantain with vegetables "
            "and a protein source such as beans, fish or chicken."
        )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return (
        "I can help with nutrition questions, meal ideas, "
        "food choices, portions, protein, fibre, hydration, "
        "weight management and Nigerian foods. For a detailed "
        "answer, describe what you are trying to achieve and "
        "include any relevant dietary restrictions or health "
        "conditions."
    )


# ============================================================
# ASK DEEPSEEK
# ============================================================

def ask_deepseek(
    question,
):

    client = get_deepseek_client()

    if client is None:

        return offline_response(
            question
        )

    profile_context = build_patient_context()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
            + profile_context,
        }
    ]

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    history = st.session_state.get(
        "nutrition_messages",
        [],
    )

    messages.extend(
        history[
            -MAX_HISTORY_MESSAGES:
        ]
    )

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:

        response = client.chat.completions.create(
            model=get_model_name(),
            messages=messages,
            stream=False,
            max_tokens=1800,
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if not answer:

            return (
                "I wasn't able to generate a response. "
                "Please try again."
            )

        return answer.strip()

    except Exception as error:

        st.warning(
            "DeepSeek could not be reached. "
            "Medusa is using its offline nutrition assistant."
        )

        with st.expander(
            "Technical information"
        ):

            st.code(
                str(error)
            )

        return offline_response(
            question
        )


# ============================================================
# CHAT MESSAGE
# ============================================================

def add_message(
    role,
    content,
):

    st.session_state.nutrition_messages.append(
        {
            "role": role,
            "content": content,
        }
    )


# ============================================================
# CHAT UI
# ============================================================

def show_nutrition_chat():

    st.subheader(
        "Ask Medusa Nutrition AI"
    )

    if deepseek_available():

        st.success(
            "DeepSeek Nutrition AI is connected."
        )

    else:

        st.info(
            "Nutrition AI is running in offline mode. "
            "Add a DEEPSEEK_API_KEY to enable DeepSeek."
        )

    # --------------------------------------------------------
    # Display previous conversation
    # --------------------------------------------------------

    for message in st.session_state.nutrition_messages:

        role = message.get(
            "role",
            "user",
        )

        content = message.get(
            "content",
            "",
        )

        if role not in [
            "user",
            "assistant",
        ]:

            continue

        with st.chat_message(
            role
        ):

            st.markdown(
                content
            )

    # --------------------------------------------------------
    # User question
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a nutrition question..."
    )

    if question:

        add_message(
            "user",
            question,
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Medusa is thinking..."
            ):

                answer = ask_deepseek(
                    question
                )

            st.markdown(
                answer
            )

        add_message(
            "assistant",
            answer,
        )

        st.rerun()


# ============================================================
# QUICK QUESTIONS
# ============================================================

def show_quick_questions():

    st.subheader(
        "Quick Nutrition Questions"
    )

    questions = [
        "Give me a healthy Nigerian breakfast.",
        "What are affordable high-protein foods?",
        "How can I eat healthier on a small budget?",
        "Create a balanced Nigerian lunch.",
        "What foods are high in fibre?",
        "How can I reduce added sugar?",
    ]

    cols = st.columns(2)

    for index, question in enumerate(
        questions
    ):

        with cols[index % 2]:

            if st.button(
                question,
                key=f"nutrition_quick_{index}",
                use_container_width=True,
            ):

                add_message(
                    "user",
                    question,
                )

                with st.spinner(
                    "Medusa is thinking..."
                ):

                    answer = ask_deepseek(
                        question
                    )

                add_message(
                    "assistant",
                    answer,
                )

                st.rerun()


# ============================================================
# DIET ASSESSMENT
# ============================================================

def calculate_nutrition_score(
    meals,
    vegetables,
    fruits,
    protein,
    water,
    processed,
):

    score = 0

    if meals >= 3:
        score += 15
    elif meals == 2:
        score += 10
    elif meals == 1:
        score += 5

    if vegetables >= 2:
        score += 20
    elif vegetables == 1:
        score += 12

    if fruits >= 2:
        score += 15
    elif fruits == 1:
        score += 10

    if protein >= 2:
        score += 20
    elif protein == 1:
        score += 12

    if water >= 4:
        score += 20
    elif water >= 2:
        score += 12
    elif water >= 1:
        score += 6

    if processed <= 1:
        score += 10
    elif processed == 2:
        score += 5

    return min(
        score,
        100,
    )


def show_diet_assessment():

    st.subheader(
        "Dietary Assessment"
    )

    st.caption(
        "This is an educational nutrition screen, "
        "not a clinical diagnosis."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        meals = st.number_input(
            "Meals per day",
            min_value=0,
            max_value=10,
            value=3,
        )

        vegetables = st.number_input(
            "Vegetable servings/day",
            min_value=0,
            max_value=10,
            value=2,
        )

    with col2:

        fruits = st.number_input(
            "Fruit servings/day",
            min_value=0,
            max_value=10,
            value=2,
        )

        protein = st.number_input(
            "Protein servings/day",
            min_value=0,
            max_value=10,
            value=2,
        )

    with col3:

        water = st.number_input(
            "Approx. cups of water/day",
            min_value=0,
            max_value=30,
            value=6,
        )

        processed = st.number_input(
            "Processed-food servings/day",
            min_value=0,
            max_value=10,
            value=1,
        )

    if st.button(
        "Calculate Nutrition Score",
        type="primary",
        use_container_width=True,
    ):

        score = calculate_nutrition_score(
            meals,
            vegetables,
            fruits,
            protein,
            water,
            processed,
        )

        st.session_state.nutrition_score = score

        st.rerun()

    score = st.session_state.get(
        "nutrition_score"
    )

    if score is not None:

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Nutrition score",
                f"{score}/100",
            )

        with col2:

            if score >= 80:

                status = "Good"

            elif score >= 60:

                status = "Needs improvement"

            else:

                status = "Needs attention"

            st.metric(
                "Overall status",
                status,
            )

        st.progress(
            score / 100
        )


# ============================================================
# MEAL PLAN GENERATOR
# ============================================================

def generate_meal_plan():

    st.subheader(
        "AI Meal Plan"
    )

    col1, col2 = st.columns(2)

    with col1:

        days = st.selectbox(
            "Plan length",
            [1, 3, 5, 7],
            index=1,
        )

        style = st.selectbox(
            "Food preference",
            [
                "Nigerian / West African",
                "General balanced diet",
                "High-protein",
                "Budget-friendly",
                "Vegetarian",
            ],
        )

    with col2:

        target = st.selectbox(
            "Goal",
            [
                "Healthy eating",
                "Weight management",
                "Muscle support",
                "Better energy",
            ],
        )

        restrictions = st.text_input(
            "Restrictions",
            placeholder=(
                "e.g. no milk, peanut allergy"
            ),
        )

    if st.button(
        "Generate Meal Plan",
        type="primary",
        use_container_width=True,
    ):

        prompt = f"""
Create a practical {days}-day meal plan.

Food style:
{style}

Goal:
{target}

Dietary restrictions:
{restrictions or "None provided"}

Patient profile:
{build_patient_context()}

For each day provide:
- Breakfast
- Lunch
- Dinner
- Optional snack

Use realistic foods and portions.

Do not prescribe a medical diet.
Flag anything that requires professional dietary assessment.
"""

        with st.spinner(
            "Creating meal plan..."
        ):

            answer = ask_deepseek(
                prompt
            )

        st.session_state.nutrition_meal_plan = answer

    plan = st.session_state.get(
        "nutrition_meal_plan"
    )

    if plan:

        st.divider()

        st.markdown(
            plan
        )


# ============================================================
# DIETITIAN REVIEW
# ============================================================

def show_dietitian_review():

    st.subheader(
        "Professional Dietitian Review"
    )

    st.write(
        "AI can help organize your nutrition information, "
        "but a qualified dietitian can provide individualized "
        "professional assessment."
    )

    if st.session_state.dietitian_review_requested:

        st.success(
            "Dietitian review request recorded."
        )

        return

    reason = st.text_area(
        "What would you like the dietitian to help with?",
        placeholder=(
            "Example: I want help creating a meal plan "
            "for my lifestyle."
        ),
    )

    urgency = st.selectbox(
        "Request type",
        [
            "General nutrition consultation",
            "Meal planning",
            "Weight management",
            "Sports nutrition",
            "Nutrition during illness",
            "Other",
        ],
    )

    if st.button(
        "Request Dietitian",
        type="primary",
        use_container_width=True,
    ):

        if not reason.strip():

            st.warning(
                "Please describe what you need help with."
            )

            return

        st.session_state.dietitian_review_requested = True

        st.success(
            "Your dietitian request has been recorded."
        )

        st.caption(
            f"Request type: {urgency}"
        )

        st.caption(
            datetime.now().strftime(
                "Submitted %Y-%m-%d at %H:%M"
            )
        )


# ============================================================
# CLEAR CHAT
# ============================================================

def clear_nutrition_chat():

    st.session_state.nutrition_messages = []

    st.success(
        "Nutrition conversation cleared."
    )

    st.rerun()


# ============================================================
# MAIN
# ============================================================

def show_dietitian():

    initialize_state()

    st.title(
        "Dietitian & Nutrition AI"
    )

    st.caption(
        "Ask questions, assess your diet, create meal plans "
        "and connect with professional nutrition support."
    )

    # --------------------------------------------------------
    # Safety notice
    # --------------------------------------------------------

    st.warning(
        "Medusa Nutrition AI provides educational guidance. "
        "It does not diagnose disease or replace a qualified "
        "dietitian or doctor."
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if deepseek_available():

        st.caption(
            f"AI engine: DeepSeek • {get_model_name()}"
        )

    else:

        st.caption(
            "AI engine: Medusa offline nutrition assistant"
        )

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    tabs = st.tabs(
        [
            "🤖 Ask Medusa",
            "👤 Nutrition Profile",
            "📊 Diet Assessment",
            "🍽️ Meal Planner",
            "🥗 Quick Questions",
            "👩‍⚕️ Dietitian",
        ]
    )

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    with tabs[0]:

        show_nutrition_chat()

        if st.session_state.nutrition_messages:

            if st.button(
                "Clear conversation",
                use_container_width=True,
            ):

                clear_nutrition_chat()

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    with tabs[1]:

        show_nutrition_profile()

    # --------------------------------------------------------
    # ASSESSMENT
    # --------------------------------------------------------

    with tabs[2]:

        show_diet_assessment()

    # --------------------------------------------------------
    # MEAL PLANNER
    # --------------------------------------------------------

    with tabs[3]:

        generate_meal_plan()

    # --------------------------------------------------------
    # QUICK QUESTIONS
    # --------------------------------------------------------

    with tabs[4]:

        show_quick_questions()

    # --------------------------------------------------------
    # DIETITIAN
    # --------------------------------------------------------

    with tabs[5]:

        show_dietitian_review()


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    show_dietitian()
