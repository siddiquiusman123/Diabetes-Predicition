import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Diabetes Prediction App")

model = joblib.load("Diabetes_prd_model.pkl")
scaler = joblib.load("Diabetes_scaler.pkl")

pregnancies = st.number_input("Pregnancies", min_value=0 , max_value=20 , step=1)
glucose = st.number_input("Glucose",min_value=20 , max_value=230)
bloodpressure = st.number_input("BloodPressure",min_value=20 , max_value=140 , step=1)
skinthickness = st.number_input("SkinThickness",min_value=3 , max_value=100)
insulin = st.number_input("Insulin",min_value=10 , max_value=1000)
bmi = st.number_input("BMI",min_value=10 , max_value=100)
diabetes_pedigree_function = st.number_input("DiabetesPedigreeFunction")
age = st.number_input("Age",min_value=0 ,max_value=100 , step=1)	

user_input = [[pregnancies, glucose, bloodpressure, skinthickness,
                insulin, bmi, diabetes_pedigree_function, age]]

columnss = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']

input_df = pd.DataFrame(user_input, columns=columnss)
numeric_colums = input_df.select_dtypes(include=['int64','float64']).columns
input_df[numeric_colums] = scaler.transform(input_df[numeric_colums])

if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.success("You have Diabetes")
    else:
        st.success("You Dont Have Diabetes")