#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:23:35 2024

@author: tynash
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Loading the saved model
diabetes_model, explainer = pickle.load(open('/home/tynash/My_Projects/Python/Diabetes_Prediction/diabetes_model_and_explainer.pkl', 'rb'))

# Navigation Side bar
with st.sidebar:
    selected = option_menu('Disease Prediction System Using ML',
                           ['Diabetes Prediction'], 
                           icons=['activity'],
                           default_index=0)
    
# Diabetes Prediction page
if selected == 'Diabetes Prediction':
    # Title page
    st.title('Diabetes Prediction Using ML')

    # Getting input data from the user
    # Columns for input data
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Gender', options=['Male', 'Female', 'Other'])
        
        age = st.number_input('Age', min_value=0)

        hypertension = st.selectbox('Hypertension', options=['Yes', 'No'])

        heart_disease = st.selectbox('Heart Disease', options=['Yes', 'No'])

    with col2:
        smoking_history = st.selectbox('Smoking History', options=['Never', 'Former', 'Current'])
        
        bmi = st.number_input('BMI', min_value=0.0, format="%.2f")

        HbA1c_level = st.number_input('HbA1c Level (%)', min_value=0.0, format="%.2f")

        blood_glucose_level = st.number_input('Blood Glucose Level (mg/dL)', min_value=0.0, format="%.2f")

    # Code for Prediction
    diab_diagnosis = ''

    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        # Encode categorical variables
        gender_encoded = 1 if gender == 'Male' else (0 if gender == 'Female' else 2)
        hypertension_encoded = 1 if hypertension == 'Yes' else 0
        heart_disease_encoded = 1 if heart_disease == 'Yes' else 0
        smoking_history_encoded = {
            'Never': 0,
            'Former': 1,
            'Current': 2
        }[smoking_history]
        
        # Prepare input for prediction
        input_data = [[gender_encoded, age, hypertension_encoded, heart_disease_encoded, smoking_history_encoded, bmi, HbA1c_level, blood_glucose_level]]
        
        # Make prediction
        diab_prediction = diabetes_model.predict(input_data)
        
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The Person is Diabetic'
        else:
            diab_diagnosis = 'The Person is not Diabetic'
        
    st.success(diab_diagnosis)