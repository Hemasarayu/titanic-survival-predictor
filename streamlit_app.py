import streamlit as st
import pandas as pd
from model import train_model, predict_survival

st.title("Titanic Survival Prediction")

uploaded_file = st.file_uploader("Upload Titanic Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    model = train_model(df)

    st.write("Enter passenger details to predict survival:")

    pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Number of siblings/spouses aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Number of parents/children aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Passenger fare", min_value=0.0, max_value=600.0, value=32.2)

    if st.button("Predict Survival"):
        passenger = {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
        }
        prediction = predict_survival(model, passenger)
        if prediction == 1:
            st.success("The passenger **survived** the Titanic disaster.")
        else:
            st.error("The passenger **did not survive** the Titanic disaster.")
