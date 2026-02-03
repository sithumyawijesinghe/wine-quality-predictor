import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🍷 Wine Quality Predictor")

st.sidebar.header("Input Features")


model_columns = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "Id"
]

default_values = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11,
    "total sulfur dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

data = {}

# Get input 
for col in model_columns:
    if col != "Id":
        data[col] = st.sidebar.number_input(col, value=float(default_values[col]))

 
data["Id"] = 1  

# Create DataFrame with exact columns order
input_df = pd.DataFrame(data, index=[0])
input_df = input_df[model_columns]

st.write("User Input:")
st.write(input_df)

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Wine Quality: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
