# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model, scaler, and encoders
MODEL_PATH = 'income_prediction_model.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODERS_PATH = 'label_encoders.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

# Set up the Streamlit app title and description
st.title('Income Prediction Model')
st.write('This app predicts annual income based on demographic and work-related data.')
st.write('---')

# Get the classes for the dropdowns from the label encoders
education_options = list(label_encoders['Education'].classes_)
occupation_options = list(label_encoders['Occupation'].classes_)

# Create input fields for user data
st.header('Enter Your Information')

age = st.number_input('Age', min_value=18, max_value=100, value=30, step=1)
hours_per_week = st.number_input('Hours Per Week', min_value=1, max_value=100, value=40, step=1)
education = st.selectbox('Education Level', options=education_options)
occupation = st.selectbox('Occupation', options=occupation_options)

# Create a button to make predictions
if st.button('Predict Income'):
    # Encode the categorical inputs
    education_encoded = label_encoders['Education'].transform([education])[0]
    occupation_encoded = label_encoders['Occupation'].transform([occupation])[0]
    
    # Create a DataFrame from the user inputs
    user_data = pd.DataFrame({
        'Age': [age],
        'Education': [education_encoded],
        'Occupation': [occupation_encoded],
        'HoursPerWeek': [hours_per_week]
    })
    
    # Scale the user data using the loaded scaler
    user_data_scaled = scaler.transform(user_data)
    
    # Make a prediction
    predicted_income = model.predict(user_data_scaled)[0]
    
    # Display the prediction
    st.success(f'Predicted Annual Income: ${predicted_income:,.2f}')

st.write('---')
st.info('**Disclaimer:** This is a simple model trained on limited data. The predictions are for demonstration purposes only.')
