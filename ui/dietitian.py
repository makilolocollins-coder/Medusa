# ============================================================
# MEDUSA AI
# DIETITIAN MODULE
#
# Purpose:
# - Patient nutrition profile
# - Dietary assessment
# - Nutrition recommendations
# - Dietitian review request
# - Simple nutrition metrics
#
# This module does NOT perform medical diagnosis.
# ============================================================

import streamlit as st
from datetime import datetime


# ============================================================
# NUTRITION DATA
# ============================================================

FOOD_GROUPS = {
    "Protein": [
        "Eggs",
        "Fish",
        "Chicken",
        "Beans",
        "Lentils",
        "Soy",
    ],
    "Vegetables": [
        "Leafy vegetables",
        "Carrots",
        "Tomatoes",
        "Okra",
        "Pepper",
        "Cucumber",
    ],
    "Fruits": [
        "Orange",
        "Banana",
        "Pawpaw",
        "Watermelon",
        "Pineapple",
        "Apple",
    ],
    "Whole grains / staples": [
        "Brown rice",
        "Oats",
        "Whole-grain bread",
        "Yam",
        "Sweet potato",
        "Maize",
    ],
    "Healthy fats": [
        "Avocado",
        "Groundnuts",
        "Cashews",
        "Sesame",
        "Olive oil",
    ],
}


# ============================================================
# SESSION STATE
# ============================================================

def _initialize_state():

    defaults = {
        "dietitian_profile": {},
        "nutrition_score": None,
        "diet_plan": [],
        "dietitian_review_requested": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# NUTRITION SCORE
# ============================================================

def calculate_nutrition_score(
    meals_per_day,
    vegetables,
    fruits,
    protein,
    water,
    processed_food,
):
    """
    Simple lifestyle-oriented nutrition score.

    This is NOT a clinical nutrition assessment.
    """

    score = 0

    # Meals
    if meals_per_day >= 3:
        score += 15
    elif meals_per_day == 2:
        score += 10
    else:
        score += 5

    # Vegetables
    if vegetables >= 2:
        score += 20
    elif vegetables == 1:
        score += 12

    # Fruits
    if fruits >= 2:
        score += 15
    elif fruits == 1:
        score += 10

    # Protein
    if protein >= 2:
        score += 20
    elif protein == 1:
        score += 12

    # Water
    if water >= 2:
        score += 20
    elif water == 1:
        score += 10

    # Processed food
    if processed_food <= 1:
        score += 10
    elif processed_food == 2:
        score += 5

    return min(score, 100)


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    meals_per_day,
    vegetables,
    fruits,
    protein,
    water,
    processed_food,
):
    recommendations = []

    if meals_per_day < 3:
        recommendations.append(
            "Consider establishing regular balanced meals "
            "throughout the day."
        )

    if vegetables < 2:
        recommendations.append(
            "Increase your intake of vegetables, especially "
            "leafy vegetables."
        )

    if fruits < 2:
        recommendations.append(
            "Consider adding more whole fruits to your diet."
        )

    if protein < 2:
        recommendations.append(
            "Include a protein source such as beans, eggs, "
            "fish or chicken in your meals."
        )

    if water < 2:
        recommendations.append(
            "Increase your water intake according to your "
            "individual needs and medical advice."
        )

    if processed_food >= 2:
        recommendations.append(
            "Try to reduce highly processed and heavily "
            "sugared foods."
        )

    if not recommendations:
        recommendations.append(
            "Your reported dietary habits look balanced. "
            "Continue maintaining variety and moderation."
        )

    return recommendations


# ============================================================
# PATIENT PROFILE
# ============================================================

def show_patient_profile():

    st.subheader("Nutrition Profile")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25,
        )

        sex = st.selectbox(
            "Sex",
            [
                "Prefer not to say",
                "Male",
                "Female",
            ],
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

        dietary_pattern = st.selectbox(
            "Dietary pattern",
            [
                "Mixed diet",
                "Vegetarian",
                "Vegan",
                "Other",
            ],
        )

        goal = st.selectbox(
            "Primary nutrition goal",
            [
                "General healthy eating",
                "Weight management",
                "Muscle support",
                "Better energy",
                "Disease-supportive nutrition",
            ],
        )

        allergies = st.text_input(
            "Known food allergies",
            placeholder="e.g. peanuts, milk",
        )

    if st.button(
        "Save Nutrition Profile",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.dietitian_profile = {
            "age": age,
            "sex": sex,
            "activity": activity,
            "dietary_pattern": dietary_pattern,
            "goal": goal,
            "allergies": allergies,
        }

        st.success(
            "Nutrition profile saved."
        )


# ============================================================
# DIETARY ASSESSMENT
# ============================================================

def show_assessment():

    st.subheader("Dietary Assessment")

    st.caption(
        "Tell Medusa about your typical daily eating habits."
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
            value=1,
        )

    with col2:

        fruits = st.number_input(
            "Fruit servings/day",
            min_value=0,
            max_value=10,
            value=1,
        )

        protein = st.number_input(
            "Protein servings/day",
            min_value=0,
            max_value=10,
            value=2,
        )

    with col3:

        water = st.number_input(
            "Water intake (rough estimate)",
            min_value=0,
            max_value=20,
            value=4,
        )

        processed = st.number_input(
            "Processed-food servings/day",
            min_value=0,
            max_value=10,
            value=1,
        )

    if st.button(
        "Analyze Diet",
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

        recommendations = generate_recommendations(
            meals,
            vegetables,
            fruits,
            protein,
            water,
            processed,
        )

        st.session_state.nutrition_score = score
        st.session_state.diet_plan = recommendations

        st.success(
            "Dietary assessment completed."
        )


# ============================================================
# RESULTS
# ============================================================

def show_results():

    score = st.session_state.get(
        "nutrition_score"
    )

    recommendations = st.session_state.get(
        "diet_plan",
        [],
    )

    if score is None:
        return

    st.divider()

    st.subheader("Nutrition Overview")

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

    st.subheader(
        "Personalized Suggestions"
    )

    for recommendation in recommendations:

        st.info(
            recommendation
        )


# ============================================================
# FOOD EXPLORER
# ============================================================

def show_food_explorer():

    st.subheader("Food Explorer")

    group = st.selectbox(
        "Select a food group",
        list(FOOD_GROUPS.keys()),
    )

    foods = FOOD_GROUPS[group]

    cols = st.columns(3)

    for index, food in enumerate(foods):

        with cols[index % 3]:

            st.markdown(
                f"**{food}**"
            )


# ============================================================
# DIETITIAN REVIEW
# ============================================================

def request_dietitian_review():

    st.subheader(
        "Dietitian Review"
    )

    st.write(
        "Want a qualified professional to review your "
        "nutrition profile?"
    )

    if st.session_state.dietitian_review_requested:

        st.success(
            "Your dietitian review request has been submitted."
        )

        st.caption(
            "A dietitian can review your dietary information "
            "and provide professional guidance."
        )

        return

    reason = st.text_area(
        "What would you like help with?",
        placeholder=(
            "Describe your nutrition goal or concern..."
        ),
    )

    priority = st.selectbox(
        "Review type",
        [
            "General nutrition guidance",
            "Meal planning",
            "Weight management",
            "Nutrition support during treatment",
        ],
    )

    if st.button(
        "Request Dietitian Review",
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
            "Dietitian review request submitted."
        )

        st.caption(
            f"Request type: {priority}"
        )

        st.caption(
            f"Submitted: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )


# ============================================================
# MAIN PAGE
# ============================================================

def show_dietitian():

    _initialize_state()

    st.title(
        "Nutrition & Dietetics"
    )

    st.caption(
        "Personalized nutrition support within Medusa AI"
    )

    st.warning(
        "Nutrition recommendations are for general guidance "
        "and do not replace assessment by a qualified dietitian "
        "or doctor."
    )

    tabs = st.tabs(
        [
            "Nutrition Profile",
            "Diet Assessment",
            "Food Explorer",
            "Dietitian Review",
        ]
    )

    with tabs[0]:

        show_patient_profile()

    with tabs[1]:

        show_assessment()

        show_results()

    with tabs[2]:

        show_food_explorer()

    with tabs[3]:

        request_dietitian_review()


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    show_dietitian()

Then make only these changes to your "app.py"

1. Add this import:

from ui.dietitian import show_dietitian

2. Add ""Nutrition"" to "patient_pages":

patient_pages = [
    "Dashboard",
    "AI Detection",
    "Examinations",
    "Reports",
    "Health",
    "Nutrition",
    "Marketplace",
    "Profile",
]

3. Add the same ""Nutrition"" entry to the radiologist "pages" list if you want everyone to see it:

pages = [
    "Dashboard",
    "AI Detection",
    "Examinations",
    "Reports",
    "Health",
    "Nutrition",
    "Marketplace",
    "Profile",
    "Radiologist",
]

4. Add this route before Marketplace:

elif selected_page == "Nutrition":

    show_dietitian()

That's the first working version. It gives Medusa a real nutrition module now, while keeping the architecture ready for the next stage: connecting patient nutrition data to Supabase, actual dietitian accounts, meal plans, appointments, and professional review.
