import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/titanic_best_model.pkl")

st.title("Titanic Survival Prediction")

st.write("Enter passenger details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

sibsp = st.number_input(
    "Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)

if st.button("Predict Survival"):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("Passenger is predicted to survive.")
    else:
        st.error("Passenger is predicted not to survive.")

    st.write(
        "Survival Probability:",
        round(probability[0][1] * 100, 2),
        "%"
    )

    st.write(
        "Non-Survival Probability:",
        round(probability[0][0] * 100, 2),
        "%"
    )