"""
Flask Application for Type 2 Diabetes Prediction with Interpretability
Author: Tinashe Jima
Reg Number: R217094J
"""

from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
import pickle
import shap
from shap import TreeExplainer
from lime import lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # For session management

# Global variables for model and explainers
model = None
explainer = None
feature_names = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 
                'blood_glucose_level', 'smoking_history_No Info', 'smoking_history_current',
                'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
                'smoking_history_not current', 'gender_Female', 'gender_Male', 'gender_Other']

def load_model():
    """Load the trained model and explainer"""
    global model, explainer
    
    try:
        # Load your existing model and explainer
        with open('rf_interpretability.pkl', 'rb') as f:
            model, explainer = pickle.load(f)
        
        print("Model and explainer loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_input(data):
    """Preprocess user input data to match model requirements"""
    
    # Create base dictionary with numerical values
    processed_data = {
        'age': float(data.get('age', 0)),
        'hypertension': int(data.get('hypertension', 0)),
        'heart_disease': int(data.get('heart_disease', 0)),
        'bmi': float(data.get('bmi', 0)),
        'HbA1c_level': float(data.get('HbA1c_level', 0)),
        'blood_glucose_level': float(data.get('blood_glucose_level', 0))
    }
    
    # Gender encoding (one-hot) - FIXED
    gender = data.get('gender', 'Female')
    # Set all gender columns to 0 first
    processed_data['gender_Female'] = 0
    processed_data['gender_Male'] = 0
    processed_data['gender_Other'] = 0
    # Then set the selected gender to 1
    if gender == 'Female':
        processed_data['gender_Female'] = 1
    elif gender == 'Male':
        processed_data['gender_Male'] = 1
    elif gender == 'Other':
        processed_data['gender_Other'] = 1
    
    # Smoking history encoding (one-hot)
    smoking_history = data.get('smoking_history', 'never')
    smoking_categories = ['No Info', 'current', 'ever', 'former', 'never', 'not current']
    
    # Set all smoking columns to 0 first
    for category in smoking_categories:
        col_name = f'smoking_history_{category.replace(" ", "_")}'
        processed_data[col_name] = 0
    
    # Then set the selected smoking history to 1
    for category in smoking_categories:
        col_name = f'smoking_history_{category.replace(" ", "_")}'
        if smoking_history.lower() == category.lower().replace("_", " "):
            processed_data[col_name] = 1
            break
    
    # Create DataFrame with correct feature order
    df = pd.DataFrame([processed_data])[feature_names]
    return df

def get_risk_recommendation(prediction, confidence):
    """Determine risk level and recommendation based on prediction and confidence"""
    if prediction == 1:
        if confidence >= 70:
            risk_level = "High Risk of Diabetes"
            recommendation = "Schedule immediate consultation"
        elif confidence >= 50:
            risk_level = "Moderate Risk of Diabetes"
            recommendation = "Schedule follow-up appointment"
        else:
            risk_level = "Low-Moderate Risk of Diabetes"
            recommendation = "Monitor and lifestyle changes"
    else:
        if confidence <= 30:
            risk_level = "Low Risk of Diabetes"
            recommendation = "Continue healthy lifestyle"
        else:
            risk_level = "Very Low Risk of Diabetes"
            recommendation = "Regular health checkups"
    
    return risk_level, recommendation

def analyze_feature_impacts(model, sample_data, explainer):
    """Analyze feature impacts using SHAP values"""
    
    # Get prediction and probability
    prediction = model.predict(sample_data)[0]
    probability = model.predict_proba(sample_data)[0][1]
    
    # Get SHAP values
    shap_values_raw = explainer.shap_values(sample_data)
    if isinstance(shap_values_raw, list):
        shap_values_raw = shap_values_raw[1]  # For binary classification
    
    shap_values = shap_values_raw[0]
    
    # Calculate feature impacts
    feature_impacts = []
    for i, (feature, value) in enumerate(sample_data.iloc[0].items()):
        impact = shap_values[i]
        feature_impacts.append({
            'feature': feature,
            'value': value,
            'shap_impact': float(impact),
            'abs_impact': abs(float(impact))
        })
    
    # Sort by absolute impact
    feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)
    
    return feature_impacts, prediction, probability

def format_feature_name(feature_name):
    """Format feature names for display"""
    # Remove prefixes
    if feature_name.startswith('smoking_history_'):
        return feature_name.replace('smoking_history_', '').replace('_', ' ').title()
    elif feature_name.startswith('gender_'):
        return feature_name.replace('gender_', '').title()
    else:
        return feature_name.replace('_', ' ').title()

def format_feature_value(feature_name, value):
    """Format feature values for display"""
    if feature_name in ['hypertension', 'heart_disease']:
        return "Yes" if value == 1 else "No"
    elif 'smoking_history' in feature_name:
        return "Yes" if value == 1 else "No"
    elif 'gender' in feature_name:
        return "Yes" if value == 1 else "No"
    else:
        return value

def calculate_risk_percentage(impact, base_probability):
    """Calculate risk percentage change for display"""
    risk_change = (impact / base_probability) * 100
    return risk_change

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Preprocess input
        sample_data = preprocess_input(data)
        
        # Analyze feature impacts
        feature_impacts, prediction, probability = analyze_feature_impacts(model, sample_data, explainer)
        confidence = probability * 100
        
        # Get risk level and recommendation
        risk_level, recommendation = get_risk_recommendation(prediction, confidence)
        
        # Prepare factors for display
        top_factors = []
        for factor in feature_impacts[:10]:  # Top 10 factors
            formatted_name = format_feature_name(factor['feature'])
            formatted_value = format_feature_value(factor['feature'], factor['value'])
            risk_pct = calculate_risk_percentage(factor['shap_impact'], probability)
            
            top_factors.append({
                'name': formatted_name,
                'value': formatted_value,
                'impact': factor['shap_impact'],
                'risk_change': risk_pct,
                'direction': 'increases' if factor['shap_impact'] > 0 else 'decreases'
            })
        
        # Store in session for result display
        session['prediction_results'] = {
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'top_factors': top_factors,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input_data': data
        }
        
        return render_template('result.html', 
                             prediction=prediction,
                             probability=probability,
                             confidence=confidence,
                             risk_level=risk_level,
                             recommendation=recommendation,
                             top_factors=top_factors,
                             input_data=data)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        sample_data = preprocess_input(data)
        
        feature_impacts, prediction, probability = analyze_feature_impacts(model, sample_data, explainer)
        confidence = probability * 100
        risk_level, recommendation = get_risk_recommendation(prediction, confidence)
        
        # Prepare factors for API response
        top_factors = []
        for factor in feature_impacts[:5]:
            formatted_name = format_feature_name(factor['feature'])
            risk_pct = calculate_risk_percentage(factor['shap_impact'], probability)
            
            top_factors.append({
                'feature': formatted_name,
                'impact': float(factor['shap_impact']),
                'risk_change_percent': float(risk_pct),
                'direction': 'increases' if factor['shap_impact'] > 0 else 'decreases'
            })
        
        response = {
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence_percent': float(confidence),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'top_factors': top_factors,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/detailed_results')
def detailed_results():
    """Display detailed results in the exact format requested"""
    results = session.get('prediction_results')
    if not results:
        return render_template('error.html', error="No prediction results found")
    
    return render_template('detailed_results.html', results=results)

@app.route('/batch_predict')
def batch_predict():
    """Render batch prediction page"""
    return render_template('batch_predict.html')

@app.route('/about')
def about():
    """Render about page"""
    return render_template('about.html')

if __name__ == '__main__':
    if load_model():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check model files.")