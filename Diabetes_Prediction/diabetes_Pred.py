#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:23:35 2024

@author: tynash
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#Loading the saved model
diabetes_model = pickle.load(open('/home/tynash/My_Projects/Python/Diabetes_Prediction/diabetes_model.sav', 'rb'))


# Navigation Side bar
with st.sidebar:
    selected = option_menu('Disease Prediction System Using ML',
                           
                           ['Diabetes Prediction', 
                          'Heart Disease Prediction'],
                           
                           icons = ['activity', 'heart'],
                           default_index = 0)
    
    
# Diabetes Prediction page


if selected == 'Diabetes Prediction':
    # title page
    st.title('Diabetes Prediction Using ML')

    # Getting input data from the user

    # Columns for input data
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI Value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')

    with col2:
        Age = st.text_input('Age of the person')


# Code for Prediction

diab_diagnosis = ''

#Creating a button for prediction
if st.button('Diabetes Test Result'):
    diab_prediction = diabetes_model.predict([[Pregnancies,Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,Age]])
    

    if diab_prediction[0] == 1:
        diab_diagnosis = 'The Person is Diabetic'
    else:
        diab_diagnosis = 'The Person is not Diabetic'
        
st.success(diab_diagnosis)        
    



   
    
    
    
    
    
if(selected == 'Heart Disease Prediction'):

   #title page
    st.title('Heart Disease Prediction Using ML') 

       
       
       