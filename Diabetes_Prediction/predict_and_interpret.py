
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import shap
import pickle

# Load the dataset
df = pd.read_csv('diabetes_dataset.csv')

# Data Preprocessing
df['smoking_history'] = df['smoking_history'].replace('No Info', 'unknown')

# Separate features and target variable
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Identify categorical and numerical features
categorical_features = ['gender', 'smoking_history']
numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define individual models
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
lr_clf = LogisticRegression(random_state=42, max_iter=1000)
svm_clf = SVC(probability=True, random_state=42)

# Create the ensemble model with a VotingClassifier
ensemble_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('lr', lr_clf), ('svm', svm_clf)],
    voting='soft'
)

# Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', ensemble_clf)])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model trained and saved as ensemble_model.pkl")

# Load the saved model
with open('ensemble_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print("Model loaded from ensemble_model.pkl")

# Create a sample data point for prediction
sample_data = pd.DataFrame({
    'gender': ['Female'],
    'age': [80.0],
    'hypertension': [0],
    'heart_disease': [1],
    'smoking_history': ['never'],
    'bmi': [25.19],
    'HbA1c_level': [6.6],
    'blood_glucose_level': [140]
})

# Make a prediction
prediction = loaded_model.predict(sample_data)
prediction_proba = loaded_model.predict_proba(sample_data)

print(f"\nPrediction for sample data: {{'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}}")
print(f"Prediction probabilities: {prediction_proba}")

# Explain the prediction with SHAP
# Create a SHAP explainer for the loaded model
explainer = shap.KernelExplainer(loaded_model.predict_proba, shap.sample(X_train, 100))

# Get the feature names after one-hot encoding
ohe_feature_names = loaded_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
feature_names = numerical_features + list(ohe_feature_names)

# Transform the sample data using the preprocessor
transformed_sample = loaded_model.named_steps['preprocessor'].transform(sample_data)

# Calculate SHAP values for the sample data
shap_values = explainer.shap_values(transformed_sample)

# Generate a force plot to explain the prediction
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], transformed_sample, feature_names=feature_names, show=False)
shap.save_html("shap_explanation.html", force_plot)


print("\nSHAP explanation for the prediction has been saved to shap_explanation.html")
