# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# import pickle
# import shap

# app = Flask(__name__)

# # Load the model
# model = pickle.load(open('ensemble_model.pkl', 'rb'))

# # Initialize SHAP explainer (do this once at startup)
# explainer = None

# def initialize_explainer():
#     """Initialize SHAP explainer with background data"""
#     global explainer
#     try:
#         # Create a small background dataset with typical values
#         background_data = pd.DataFrame({
#             'age': [40, 50, 60, 30, 70],
#             'hypertension': [0, 1, 0, 0, 1],
#             'heart_disease': [0, 0, 1, 0, 1],
#             'bmi': [25, 30, 28, 22, 32],
#             'HbA1c_level': [5.5, 6.0, 5.8, 5.2, 6.5],
#             'blood_glucose_level': [100, 120, 110, 90, 140],
#             'smoking_history_No Info': [0, 0, 0, 1, 0],
#             'smoking_history_current': [0, 1, 0, 0, 0],
#             'smoking_history_ever': [0, 0, 0, 0, 0],
#             'smoking_history_former': [0, 0, 1, 0, 0],
#             'smoking_history_never': [1, 0, 0, 0, 1],
#             'smoking_history_not current': [0, 0, 0, 0, 0],
#             'gender_Female': [1, 0, 1, 0, 1],
#             'gender_Male': [0, 1, 0, 1, 0],
#             'gender_Other': [0, 0, 0, 0, 0]
#         })

#         # Feature column order expected by the model (keep in sync)
#         feature_columns = [
#             'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
#             'blood_glucose_level', 'smoking_history_No Info', 'smoking_history_current',
#             'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
#             'smoking_history_not current', 'gender_Female', 'gender_Male', 'gender_Other'
#         ]

#         # If the model is a tree-based ensemble supported by TreeExplainer, use it.
#         # Otherwise fall back to KernelExplainer which works for any estimator.
#         try:
#             explainer = shap.TreeExplainer(model, background_data)
#             print("SHAP TreeExplainer initialized successfully")
#         except Exception:
#             # KernelExplainer expects a prediction function that returns a 1D array of probabilities
#             def predict_proba_pos(x):
#                 # x is numpy array; convert to DataFrame with correct columns
#                 df_x = pd.DataFrame(x, columns=feature_columns)
#                 return model.predict_proba(df_x)[:, 1]

#             # Use a small background sample (numpy) for KernelExplainer
#             explainer = shap.KernelExplainer(predict_proba_pos, background_data[feature_columns].values)
#             print("SHAP KernelExplainer initialized (fallback)")

#     except Exception as e:
#         print(f"Error initializing SHAP explainer: {str(e)}")
#         raise

# def encode_features(data):
#     """Encode categorical features to match the model's expected format"""
#     encoded_data = {
#         'age': float(data['age']),
#         'hypertension': int(data['hypertension']),
#         'heart_disease': int(data['heart_disease']),
#         'bmi': float(data['bmi']),
#         'HbA1c_level': float(data['HbA1c_level']),
#         'blood_glucose_level': float(data['blood_glucose_level']),
#         'smoking_history_No Info': 0,
#         'smoking_history_current': 0,
#         'smoking_history_ever': 0,
#         'smoking_history_former': 0,
#         'smoking_history_never': 0,
#         'smoking_history_not current': 0,
#         'gender_Female': 0,
#         'gender_Male': 0,
#         'gender_Other': 0
#     }
    
#     # Set the appropriate smoking history column to 1
#     smoking_col = f"smoking_history_{data['smoking_history']}"
#     if smoking_col in encoded_data:
#         encoded_data[smoking_col] = 1
    
#     # Set the appropriate gender column to 1
#     gender_col = f"gender_{data['gender']}"
#     if gender_col in encoded_data:
#         encoded_data[gender_col] = 1
    
#     return encoded_data

# def get_feature_importance_explanation(df, shap_values):
#     """
#     Get top factors influencing the prediction with readable names using SHAP values
#     """
#     # Handle SHAP values based on output format
#     print(f"SHAP values type: {type(shap_values)}")
#     print(f"SHAP values shape/length: {shap_values.shape if hasattr(shap_values, 'shape') else len(shap_values)}")
    
#     # For TreeExplainer with probability output, we get an array directly
#     if isinstance(shap_values, np.ndarray):
#         # Check if it's 2D (single prediction with multiple features)
#         if len(shap_values.shape) == 2:
#             shap_vals = shap_values[0]  # Get first row
#         else:
#             shap_vals = shap_values
#     elif isinstance(shap_values, list):
#         # If it's a list (for binary classification without probability output)
#         if len(shap_values) == 2:
#             # Get SHAP values for positive class (diabetes = 1)
#             if isinstance(shap_values[1], np.ndarray):
#                 shap_vals = shap_values[1][0] if len(shap_values[1].shape) > 1 else shap_values[1]
#             else:
#                 shap_vals = shap_values[1]
#         else:
#             shap_vals = shap_values[0][0] if len(shap_values[0].shape) > 1 else shap_values[0]
#     else:
#         # Fallback: convert to numpy array
#         shap_vals = np.array(shap_values).flatten()
    
#     print(f"Processed SHAP values shape: {shap_vals.shape}")
#     print(f"Number of features in df: {len(df.columns)}")
    
#     # Create a mapping of encoded features to readable names
#     feature_names = {
#         'age': 'Age',
#         'hypertension': 'Hypertension',
#         'heart_disease': 'Heart Disease',
#         'bmi': 'BMI',
#         'HbA1c_level': 'HbA1c Level',
#         'blood_glucose_level': 'Blood Glucose Level',
#         'smoking_history_No Info': 'Smoking History: No Info',
#         'smoking_history_current': 'Smoking History: Current',
#         'smoking_history_ever': 'Smoking History: Ever',
#         'smoking_history_former': 'Smoking History: Former',
#         'smoking_history_never': 'Smoking History: Never',
#         'smoking_history_not current': 'Smoking History: Not Current',
#         'gender_Female': 'Gender: Female',
#         'gender_Male': 'Gender: Male',
#         'gender_Other': 'Gender: Other'
#     }
    
#     # Get feature names and their SHAP values
#     features = df.columns.tolist()
#     feature_values = df.iloc[0].values
    
#     # Combine features with their SHAP values and actual values
#     importance_data = []
#     for i, feature in enumerate(features):
#         shap_value = shap_vals[i] if i < len(shap_vals) else 0
        
#         importance_data.append({
#             'feature': feature_names.get(feature, feature),
#             'shap_value': abs(shap_value),
#             'actual_shap': shap_value,
#             'value': feature_values[i],
#             'impact': 'Increases' if shap_value > 0 else 'Decreases'
#         })
    
#     # Sort by absolute SHAP value
#     importance_data.sort(key=lambda x: x['shap_value'], reverse=True)
    
#     # Get top 5 factors
#     top_factors = []
#     for item in importance_data[:5]:
#         if item['shap_value'] > 0.001:  # Only include meaningful contributions
#             # Format the value display
#             if item['value'] in [0, 1]:
#                 value_display = 'Yes' if item['value'] == 1 else 'No'
#             else:
#                 value_display = f"{item['value']:.2f}"
            
#             top_factors.append({
#                 'feature': item['feature'],
#                 'value': value_display,
#                 'impact': item['impact'],
#                 'importance': round(item['shap_value'] * 100, 2)
#             })
    
#     return top_factors

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Initialize explainer if not already done
#         if explainer is None:
#             initialize_explainer()
        
#         # Get form data
#         data = {
#             'age': request.form['age'],
#             'gender': request.form['gender'],
#             'bmi': request.form['bmi'],
#             'HbA1c_level': request.form['HbA1c_level'],
#             'blood_glucose_level': request.form['blood_glucose_level'],
#             'smoking_history': request.form['smoking_history'],
#             'hypertension': request.form['hypertension'],
#             'heart_disease': request.form['heart_disease']
#         }
        
#         print("Received data:", data)
        
#         # Encode features
#         encoded_data = encode_features(data)
        
#         # Convert to DataFrame with correct column order
#         feature_columns = [
#             'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
#             'blood_glucose_level', 'smoking_history_No Info', 'smoking_history_current',
#             'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
#             'smoking_history_not current', 'gender_Female', 'gender_Male', 'gender_Other'
#         ]
        
#         df = pd.DataFrame([encoded_data], columns=feature_columns)
        
#         print("DataFrame for prediction:")
#         print(df)
        
#         # Make prediction
#         prediction = model.predict(df)[0]
#         probability = model.predict_proba(df)[0][1] * 100
        
#         # Calculate SHAP values for interpretation
#         try:
#             # Get SHAP values - handle different explainer types
#             if isinstance(explainer, shap.KernelExplainer):
#                 # KernelExplainer expects numpy input; nsamples reduces compute cost
#                 shap_values = explainer.shap_values(df.values, nsamples=100)
#             else:
#                 # TreeExplainer or other explainers
#                 shap_values = explainer.shap_values(df)
            
#             # Debug: Print SHAP values structure
#             print(f"SHAP values type: {type(shap_values)}")
#             if isinstance(shap_values, list):
#                 print(f"SHAP values is a list with {len(shap_values)} elements")
#                 for i, sv in enumerate(shap_values):
#                     print(f"  Element {i} shape: {sv.shape if hasattr(sv, 'shape') else 'N/A'}")
#             else:
#                 print(f"SHAP values shape: {shap_values.shape}")
            
#             top_factors = get_feature_importance_explanation(df, shap_values)
#             print(f"Top factors calculated successfully: {len(top_factors)} factors")
#         except Exception as shap_error:
#             print(f"SHAP calculation error: {str(shap_error)}")
#             import traceback
#             traceback.print_exc()
#             # Fallback to empty factors if SHAP fails
#             top_factors = []
        
#         print(f"Prediction: {prediction}, Probability: {probability}")
        
#         # Calculate risk level
#         if probability < 20:
#             risk_level = "Low"
#             risk_color = "#28a745"
#         elif probability < 50:
#             risk_level = "Moderate"
#             risk_color = "#ffc107"
#         else:
#             risk_level = "High"
#             risk_color = "#dc3545"
        
#         return jsonify({
#             'success': True,
#             'prediction': int(prediction),
#             'probability': round(probability, 2),
#             'risk_level': risk_level,
#             'risk_color': risk_color,
#             'top_factors': top_factors
#         })
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import shap

app = Flask(__name__)

# Load the model
model = pickle.load(open('ensemble_model.pkl', 'rb'))

# Initialize SHAP explainer (do this once at startup)
explainer = None

def initialize_explainer():
    """Initialize SHAP explainer with background data"""
    global explainer
    try:
        # Create a small background dataset with typical values
        background_data = pd.DataFrame({
            'age': [40, 50, 60, 30, 70],
            'hypertension': [0, 1, 0, 0, 1],
            'heart_disease': [0, 0, 1, 0, 1],
            'bmi': [25, 30, 28, 22, 32],
            'HbA1c_level': [5.5, 6.0, 5.8, 5.2, 6.5],
            'blood_glucose_level': [100, 120, 110, 90, 140],
            'smoking_history_No Info': [0, 0, 0, 1, 0],
            'smoking_history_current': [0, 1, 0, 0, 0],
            'smoking_history_ever': [0, 0, 0, 0, 0],
            'smoking_history_former': [0, 0, 1, 0, 0],
            'smoking_history_never': [1, 0, 0, 0, 1],
            'smoking_history_not current': [0, 0, 0, 0, 0],
            'gender_Female': [1, 0, 1, 0, 1],
            'gender_Male': [0, 1, 0, 1, 0],
            'gender_Other': [0, 0, 0, 0, 0]
        })

        # Feature column order expected by the model (keep in sync)
        feature_columns = [
            'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
            'blood_glucose_level', 'smoking_history_No Info', 'smoking_history_current',
            'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
            'smoking_history_not current', 'gender_Female', 'gender_Male', 'gender_Other'
        ]

        # If the model is a tree-based ensemble supported by TreeExplainer, use it.
        # Otherwise fall back to KernelExplainer which works for any estimator.
        try:
            explainer = shap.TreeExplainer(model, background_data)
            print("SHAP TreeExplainer initialized successfully")
        except Exception:
            # KernelExplainer expects a prediction function that returns a 1D array of probabilities
            def predict_proba_pos(x):
                # x is numpy array; convert to DataFrame with correct columns
                df_x = pd.DataFrame(x, columns=feature_columns)
                return model.predict_proba(df_x)[:, 1]

            # Use a small background sample (numpy) for KernelExplainer
            explainer = shap.KernelExplainer(predict_proba_pos, background_data[feature_columns].values)
            print("SHAP KernelExplainer initialized (fallback)")

    except Exception as e:
        print(f"Error initializing SHAP explainer: {str(e)}")
        raise

def encode_features(data):
    """Encode categorical features to match the model's expected format"""
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

def get_detailed_factor_explanation(feature_name, value, shap_value, impact):
    """
    Provide detailed medical context and explanation for each factor
    """
    explanations = {
        'Age': {
            'context': f"Your age is {value} years.",
            'interpretation': lambda v, sv: (
                f"Age is a significant risk factor for diabetes. People over 45 have higher risk. "
                f"{'Your age increases diabetes risk as metabolic function naturally declines with age.' if float(v) > 45 else 'Your relatively younger age is a protective factor.'}"
            ),
            'healthy_range': 'Risk increases significantly after age 45',
            'recommendation': lambda v: (
                'Regular screening is recommended after age 45, or earlier if other risk factors are present.' if float(v) > 45 
                else 'Maintain healthy lifestyle habits to prevent future risk.'
            )
        },
        'BMI': {
            'context': f"Your Body Mass Index (BMI) is {value}.",
            'interpretation': lambda v, sv: (
                f"{'Obesity (BMI ≥ 30) significantly increases diabetes risk due to insulin resistance.' if float(v) >= 30 else ''}"
                f"{'Being overweight (BMI 25-29.9) moderately increases diabetes risk.' if 25 <= float(v) < 30 else ''}"
                f"{'Your BMI is in the healthy range, which is protective against diabetes.' if 18.5 <= float(v) < 25 else ''}"
                f"{'Being underweight may indicate other health concerns.' if float(v) < 18.5 else ''}"
            ),
            'healthy_range': '18.5 - 24.9 (Normal weight)',
            'recommendation': lambda v: (
                'Weight loss of 5-10% can significantly reduce diabetes risk. Focus on diet and exercise.' if float(v) >= 30
                else 'Aim to reach a BMI below 25 through balanced nutrition and regular physical activity.' if float(v) >= 25
                else 'Maintain your healthy weight through continued healthy habits.'
            )
        },
        'HbA1c Level': {
            'context': f"Your HbA1c level is {value}%.",
            'interpretation': lambda v, sv: (
                f"{'HbA1c ≥ 6.5% indicates diabetes. This is your average blood sugar over 2-3 months.' if float(v) >= 6.5 else ''}"
                f"{'HbA1c 5.7-6.4% indicates prediabetes. You have elevated blood sugar levels.' if 5.7 <= float(v) < 6.5 else ''}"
                f"{'Your HbA1c is in the normal range, indicating good blood sugar control.' if float(v) < 5.7 else ''}"
            ),
            'healthy_range': 'Below 5.7% (Normal), 5.7-6.4% (Prediabetes), ≥6.5% (Diabetes)',
            'recommendation': lambda v: (
                'Immediate medical consultation needed. You may already have diabetes.' if float(v) >= 6.5
                else 'Lifestyle changes crucial: reduce sugar intake, increase exercise, and monitor regularly.' if float(v) >= 5.7
                else 'Continue maintaining healthy blood sugar through balanced diet and exercise.'
            )
        },
        'Blood Glucose Level': {
            'context': f"Your blood glucose level is {value} mg/dL.",
            'interpretation': lambda v, sv: (
                f"{'Fasting glucose ≥ 126 mg/dL indicates diabetes.' if float(v) >= 126 else ''}"
                f"{'Fasting glucose 100-125 mg/dL indicates prediabetes (impaired fasting glucose).' if 100 <= float(v) < 126 else ''}"
                f"{'Your fasting glucose is in the normal range.' if float(v) < 100 else ''}"
            ),
            'healthy_range': 'Below 100 mg/dL (fasting), 100-125 (Prediabetes), ≥126 (Diabetes)',
            'recommendation': lambda v: (
                'Medical attention required. Confirm diagnosis with additional tests.' if float(v) >= 126
                else 'Reduce simple carbohydrates, increase fiber intake, and exercise regularly.' if float(v) >= 100
                else 'Maintain healthy blood sugar with a balanced diet and regular activity.'
            )
        },
        'Hypertension': {
            'context': f"Hypertension status: {'Yes' if value == 'Yes' else 'No'}",
            'interpretation': lambda v, sv: (
                "High blood pressure often coexists with diabetes (metabolic syndrome). "
                "Both conditions can damage blood vessels and increase cardiovascular risk." if v == 'Yes'
                else "No hypertension is a positive factor for your overall metabolic health."
            ),
            'healthy_range': 'Blood pressure < 120/80 mmHg',
            'recommendation': lambda v: (
                'Manage blood pressure through medication (if prescribed), low sodium diet, and stress reduction.' if v == 'Yes'
                else 'Continue monitoring blood pressure and maintain heart-healthy lifestyle.'
            )
        },
        'Heart Disease': {
            'context': f"Heart disease history: {'Yes' if value == 'Yes' else 'No'}",
            'interpretation': lambda v, sv: (
                "Heart disease and diabetes share common risk factors and often occur together. "
                "Having heart disease increases the likelihood of insulin resistance." if v == 'Yes'
                else "No heart disease is favorable for your cardiovascular health."
            ),
            'healthy_range': 'No cardiovascular disease',
            'recommendation': lambda v: (
                'Cardiovascular care is critical. Work closely with your cardiologist and consider diabetes screening.' if v == 'Yes'
                else 'Maintain heart health through regular exercise, healthy diet, and stress management.'
            )
        },
        'Smoking History: Current': {
            'context': "You are a current smoker.",
            'interpretation': lambda v, sv: (
                "Smoking increases diabetes risk by 30-40%. It causes insulin resistance and inflammation, "
                "damaging blood vessels and impairing glucose metabolism." if v == 'Yes' else ""
            ),
            'healthy_range': 'Non-smoker',
            'recommendation': lambda v: (
                'Quitting smoking is one of the best things you can do for diabetes prevention and overall health. '
                'Seek smoking cessation programs.' if v == 'Yes' else ''
            )
        },
        'Smoking History: Former': {
            'context': "You are a former smoker.",
            'interpretation': lambda v, sv: (
                "Former smokers have reduced risk compared to current smokers, but some increased risk remains. "
                "The risk decreases over time after quitting." if v == 'Yes' else ""
            ),
            'healthy_range': 'Non-smoker',
            'recommendation': lambda v: (
                'Great job quitting! Continue to avoid tobacco and focus on other healthy lifestyle factors.' if v == 'Yes' else ''
            )
        },
        'Smoking History: Never': {
            'context': "You have never smoked.",
            'interpretation': lambda v, sv: (
                "Never smoking is a strong protective factor against diabetes and many other diseases." if v == 'Yes' else ""
            ),
            'healthy_range': 'Non-smoker',
            'recommendation': lambda v: (
                'Continue to avoid tobacco products.' if v == 'Yes' else ''
            )
        },
        'Gender: Male': {
            'context': "Your gender is Male.",
            'interpretation': lambda v, sv: (
                "Men tend to develop diabetes at lower BMI levels than women and may have different risk patterns." if v == 'Yes' else ""
            ),
            'healthy_range': 'N/A',
            'recommendation': lambda v: (
                'Men should be particularly vigilant about abdominal obesity and regular screening.' if v == 'Yes' else ''
            )
        },
        'Gender: Female': {
            'context': "Your gender is Female.",
            'interpretation': lambda v, sv: (
                "Women with history of gestational diabetes or PCOS have increased diabetes risk. "
                "Hormonal changes during menopause can also affect risk." if v == 'Yes' else ""
            ),
            'healthy_range': 'N/A',
            'recommendation': lambda v: (
                'Be aware of gender-specific risk factors like gestational diabetes and PCOS.' if v == 'Yes' else ''
            )
        },
    }
    
    # Get explanation for this feature
    explanation_data = explanations.get(feature_name, {
        'context': f"{feature_name}: {value}",
        'interpretation': lambda v, sv: f"This factor {'increases' if shap_value > 0 else 'decreases'} your diabetes risk.",
        'healthy_range': 'N/A',
        'recommendation': lambda v: 'Consult with healthcare provider for personalized advice.'
    })
    
    return {
        'context': explanation_data['context'],
        'interpretation': explanation_data['interpretation'](value, shap_value),
        'healthy_range': explanation_data['healthy_range'],
        'recommendation': explanation_data['recommendation'](value)
    }

def get_feature_importance_explanation(df, shap_values):
    """
    Get top factors influencing the prediction with readable names using SHAP values
    """
    # Handle SHAP values based on output format
    print(f"SHAP values type: {type(shap_values)}")
    print(f"SHAP values shape/length: {shap_values.shape if hasattr(shap_values, 'shape') else len(shap_values)}")
    
    # For TreeExplainer with probability output, we get an array directly
    if isinstance(shap_values, np.ndarray):
        # Check if it's 2D (single prediction with multiple features)
        if len(shap_values.shape) == 2:
            shap_vals = shap_values[0]  # Get first row
        else:
            shap_vals = shap_values
    elif isinstance(shap_values, list):
        # If it's a list (for binary classification without probability output)
        if len(shap_values) == 2:
            # Get SHAP values for positive class (diabetes = 1)
            if isinstance(shap_values[1], np.ndarray):
                shap_vals = shap_values[1][0] if len(shap_values[1].shape) > 1 else shap_values[1]
            else:
                shap_vals = shap_values[1]
        else:
            shap_vals = shap_values[0][0] if len(shap_values[0].shape) > 1 else shap_values[0]
    else:
        # Fallback: convert to numpy array
        shap_vals = np.array(shap_values).flatten()
    
    print(f"Processed SHAP values shape: {shap_vals.shape}")
    print(f"Number of features in df: {len(df.columns)}")
    
    # Create a mapping of encoded features to readable names
    feature_names = {
        'age': 'Age',
        'hypertension': 'Hypertension',
        'heart_disease': 'Heart Disease',
        'bmi': 'BMI',
        'HbA1c_level': 'HbA1c Level',
        'blood_glucose_level': 'Blood Glucose Level',
        'smoking_history_No Info': 'Smoking History: No Info',
        'smoking_history_current': 'Smoking History: Current',
        'smoking_history_ever': 'Smoking History: Ever',
        'smoking_history_former': 'Smoking History: Former',
        'smoking_history_never': 'Smoking History: Never',
        'smoking_history_not current': 'Smoking History: Not Current',
        'gender_Female': 'Gender: Female',
        'gender_Male': 'Gender: Male',
        'gender_Other': 'Gender: Other'
    }
    
    # Get feature names and their SHAP values
    features = df.columns.tolist()
    feature_values = df.iloc[0].values
    
    # Combine features with their SHAP values and actual values
    importance_data = []
    for i, feature in enumerate(features):
        shap_value = shap_vals[i] if i < len(shap_vals) else 0
        
        importance_data.append({
            'feature': feature_names.get(feature, feature),
            'shap_value': abs(shap_value),
            'actual_shap': shap_value,
            'value': feature_values[i],
            'impact': 'Increases' if shap_value > 0 else 'Decreases'
        })
    
    # Sort by absolute SHAP value
    importance_data.sort(key=lambda x: x['shap_value'], reverse=True)
    
    # Get top 5 factors with detailed explanations
    top_factors = []
    for item in importance_data[:5]:
        if item['shap_value'] > 0.001:  # Only include meaningful contributions
            # Format the value display
            if item['value'] in [0, 1]:
                value_display = 'Yes' if item['value'] == 1 else 'No'
            else:
                value_display = f"{item['value']:.2f}"
            
            # Get detailed explanation
            detailed_explanation = get_detailed_factor_explanation(
                item['feature'], 
                value_display, 
                item['actual_shap'],
                item['impact']
            )
            
            top_factors.append({
                'feature': item['feature'],
                'value': value_display,
                'impact': item['impact'],
                'importance': round(item['shap_value'] * 100, 2),
                'context': detailed_explanation['context'],
                'interpretation': detailed_explanation['interpretation'],
                'healthy_range': detailed_explanation['healthy_range'],
                'recommendation': detailed_explanation['recommendation']
            })
    
    return top_factors

def generate_overall_summary(top_factors, probability, risk_level):
    """
    Generate an overall summary of the patient's risk profile
    """
    summary = {
        'risk_assessment': '',
        'key_concerns': [],
        'positive_factors': [],
        'action_items': []
    }
    
    # Risk assessment based on probability
    if probability < 20:
        summary['risk_assessment'] = (
            f"Your diabetes risk is currently LOW ({probability:.1f}%). "
            "However, maintaining healthy lifestyle habits is crucial for prevention."
        )
    elif probability < 50:
        summary['risk_assessment'] = (
            f"Your diabetes risk is MODERATE ({probability:.1f}%). "
            "You have some risk factors that need attention. Early intervention can prevent progression."
        )
    else:
        summary['risk_assessment'] = (
            f"Your diabetes risk is HIGH ({probability:.1f}%). "
            "Multiple risk factors are present. Immediate lifestyle changes and medical consultation are strongly recommended."
        )
    
    # Analyze top factors
    for factor in top_factors:
        if factor['impact'] == 'Increases':
            summary['key_concerns'].append({
                'factor': factor['feature'],
                'reason': factor['interpretation']
            })
        else:
            summary['positive_factors'].append({
                'factor': factor['feature'],
                'reason': factor['interpretation']
            })
        
        # Add actionable recommendations
        if factor['recommendation'] and factor['recommendation'].strip():
            summary['action_items'].append({
                'factor': factor['feature'],
                'action': factor['recommendation']
            })
    
    return summary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Initialize explainer if not already done
        if explainer is None:
            initialize_explainer()
        
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
        
        print("Received data:", data)
        
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
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] * 100
        
        # Calculate SHAP values for interpretation
        try:
            # Get SHAP values - handle different explainer types
            if isinstance(explainer, shap.KernelExplainer):
                # KernelExplainer expects numpy input; nsamples reduces compute cost
                shap_values = explainer.shap_values(df.values, nsamples=100)
            else:
                # TreeExplainer or other explainers
                shap_values = explainer.shap_values(df)
            
            # Debug: Print SHAP values structure
            print(f"SHAP values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"SHAP values is a list with {len(shap_values)} elements")
                for i, sv in enumerate(shap_values):
                    print(f"  Element {i} shape: {sv.shape if hasattr(sv, 'shape') else 'N/A'}")
            else:
                print(f"SHAP values shape: {shap_values.shape}")
            
            top_factors = get_feature_importance_explanation(df, shap_values)
            print(f"Top factors calculated successfully: {len(top_factors)} factors")
        except Exception as shap_error:
            print(f"SHAP calculation error: {str(shap_error)}")
            import traceback
            traceback.print_exc()
            # Fallback to empty factors if SHAP fails
            top_factors = []
        
        print(f"Prediction: {prediction}, Probability: {probability}")
        
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
        
        # Generate overall summary
        overall_summary = generate_overall_summary(top_factors, probability, risk_level)
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': round(probability, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'top_factors': top_factors,
            'summary': overall_summary
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)