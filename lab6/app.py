import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from genetic_module import genetic_algorithm_optimized_indexed
import json
import ast

# ----------- HealthTargetAgent -----------
def health_target_agent(weight, height, age, sex, activity_level):
    if sex.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    tdee = bmr * activity_level

    calories = round(tdee)
    protein = round(0.2 * calories / 4)  # 20% калорій на білки
    fat     = round(0.25 * calories / 9) # 25% калорій на жири
    carbs   = round((calories - (protein * 4 + fat * 9)) / 4)  # решта на вуглеводи

    return {
        "Calories": calories,
        "ProteinContent": protein,
        "FatContent": fat,
        "CarbohydrateContent": carbs
    }

# ----------- Load Recipes -----------
@st.cache_data
def load_data():
    recipes = pd.read_csv("IntelligentComputing/lab6/data/recipes_cleaned.csv")
    recipes['RecipeIngredientParts'] = recipes['RecipeIngredientParts'].apply(lambda x: ast.literal_eval(x))
    return recipes

df = load_data()
recipe_list = df.to_dict(orient='records')

st.title("🍽️ Генетичний підбір рецептів на день")

# ----------- Ingredients Input -----------
ingredients_input = st.text_input("🧂 Наявні інгредієнти (через кому)", "tomato, cheese, chicken")
available_ingredients = [x.strip() for x in ingredients_input.split(",") if x.strip()]

# ----------- Time Limit -----------
max_total_time = st.number_input("⏱️ Максимальний час на приготування (хв)", min_value=10, max_value=300, value=90)

# ----------- User Profile Input for Auto Nutrition -----------
st.subheader("👤 Дані користувача для автоматичного розрахунку нутрієнтів")
col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Стать", ["male", "female"])
    age = st.number_input("Вік", min_value=10, max_value=100, value=25)
    weight = st.number_input("Вага (кг)", min_value=30.0, max_value=200.0, value=70.0)
with col2:
    height = st.number_input("Ріст (см)", min_value=120.0, max_value=220.0, value=175.0)
    activity_options = {
    "Мінімальний (сидячий)": 1.2,
    "Легкий (1–2 тренування/тиждень)": 1.375,
    "Середній (3–5 разів/тиждень)": 1.55,
    "Інтенсивний (щодня)": 1.725
    }
    activity_label = st.selectbox("Рівень активності", list(activity_options.keys()))
    activity_value = activity_options[activity_label]


# ----------- Nutrition Targets -----------
st.subheader("🍎 Цільові значення нутрієнтів на день")

use_auto = st.checkbox("🔁 Згенерувати нутрієнти автоматично з профілю", value=True)

if use_auto:
    nutrition_targets = health_target_agent(weight, height, age, sex, activity_value)
    st.success("🎯 Автоматично згенеровані цілі:")
    st.json(nutrition_targets)
else:
    nutrition_targets = {}
    for key in ["Calories", "FatContent", "CarbohydrateContent", "ProteinContent"]:
        nutrition_targets[key] = st.number_input(f"{key}", min_value=0.0, value=100.0)

# ----------- Run Genetic Algorithm -----------
if st.button("🔍 Підібрати рецепти"):
    with st.spinner("Обчислюємо..."):
        best_recipes, history = genetic_algorithm_optimized_indexed(
            recipe_list, available_ingredients, max_total_time, nutrition_targets,
            population_size=30, generations=100, mutation_rate=0.7,
            min_recipes=2, max_recipes=5
        )

    st.success("✅ Найкращий набір рецептів:")
    for i, r in enumerate(best_recipes, 1):
        st.markdown(f"### 🍽️ Рецепт {i}: {r['Name']}")
        st.write("⏱️ Час:", r['TotalTimeMinutes'], "хв")
        st.write("⭐ Рейтинг:", r.get("AggregatedRating", "N/A"))
        st.write("🧂 Інгредієнти:", ", ".join(r['RecipeIngredientParts']))

    # ----------- Нутрієнти кожної страви + аналіз -----------

    st.subheader("📊 Аналіз нутрієнтів")

    nutrients = ["Calories", "ProteinContent", "FatContent", "CarbohydrateContent"]
    rows = []

    for r in best_recipes:
        row = {"Назва": r["Name"]}
        for n in nutrients:
            row[n] = r.get(n, 0)
        rows.append(row)

    df_nutri = pd.DataFrame(rows)

    total_row = {"Назва": "Сума"}
    for n in nutrients:
        total_row[n] = df_nutri[n].sum()
    df_nutri.loc[len(df_nutri)] = total_row

    target_row = {"Назва": "Ціль"}
    for n in nutrients:
        target_row[n] = nutrition_targets.get(n, 0)
    df_nutri.loc[len(df_nutri)] = target_row

    diff_row = {"Назва": "Відхилення"}
    for n in nutrients:
        diff = df_nutri.loc[df_nutri["Назва"] == "Сума", n].values[0] - df_nutri.loc[df_nutri["Назва"] == "Ціль", n].values[0]
        diff_row[n] = round(diff, 2)
    df_nutri.loc[len(df_nutri)] = diff_row

    # Вивід
    st.dataframe(
        df_nutri.style.format({col: "{:.2f}" for col in nutrients}),
        use_container_width=True
    )

    # ----------- Fitness History Plot -----------
    st.subheader("📉 Еволюція найкращого рішення")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Покоління")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness-історія")
    st.pyplot(fig)

