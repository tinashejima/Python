from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestClassifier 

app = Flask(__name__)

model = pickle.load(open('rf_interpretability.pkl', 'rb'))

def encode_features(data):
    """
    Encode categorical features to match the model's expected format
    """
    # Initialize all encoded columns with 0
    encoded_data = {
        'age': float(data['age']),
        'hypertension': int(data['hypertension']),
        'heart_disease': int(data['heart_disease']),
        'bmi': float(data['bmi']),
        'HbA1c_level': float(data['HbA1c_level']),
        'blood_glucose_level': float(data['blood_glucose_level']),
        'smoking_history_No Info': 0,
        'smoking_history_current': 0,
        'smoking_history_ever': 0,
        'smoking_history_former': 0,
        'smoking_history_never': 0,
        'smoking_history_not current': 0,
        'gender_Female': 0,
        'gender_Male': 0,
        'gender_Other': 0
    }
    
    # Set the appropriate smoking history column to 1
    smoking_col = f"smoking_history_{data['smoking_history']}"
    if smoking_col in encoded_data:
        encoded_data[smoking_col] = 1
    
    # Set the appropriate gender column to 1
    gender_col = f"gender_{data['gender']}"
    if gender_col in encoded_data:
        encoded_data[gender_col] = 1
    
    return encoded_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'age': request.form['age'],
            'gender': request.form['gender'],
            'bmi': request.form['bmi'],
            'HbA1c_level': request.form['HbA1c_level'],
            'blood_glucose_level': request.form['blood_glucose_level'],
            'smoking_history': request.form['smoking_history'],
            'hypertension': request.form['hypertension'],
            'heart_disease': request.form['heart_disease']
        }
        
        print("Received data:", data)  # Debug print
        
        # Encode features
        encoded_data = encode_features(data)
        
        # Convert to DataFrame with correct column order
        feature_columns = [
            'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
            'blood_glucose_level', 'smoking_history_No Info', 'smoking_history_current',
            'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
            'smoking_history_not current', 'gender_Female', 'gender_Male', 'gender_Other'
        ]
        
        df = pd.DataFrame([encoded_data], columns=feature_columns)
        
        print("DataFrame for prediction:")
        print(df)
        
        # Make actual prediction using the model
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] * 100
        
        print(f"Prediction: {prediction}, Probability: {probability}")  # Debug print
        
        # Calculate risk level
        if probability < 20:
            risk_level = "Low"
            risk_color = "#28a745"
        elif probability < 50:
            risk_level = "Moderate"
            risk_color = "#ffc107"
        else:
            risk_level = "High"
            risk_color = "#dc3545"
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': round(probability, 2),
            'risk_level': risk_level,
            'risk_color': risk_color
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)