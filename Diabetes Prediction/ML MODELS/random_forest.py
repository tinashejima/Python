import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Step 1: Generate Fictitious Dataset
np.random.seed(42)
data = {
    'HIV_Status': np.random.choice(['Yes', 'No'], size=100),
    'Age': np.random.randint(20, 60, size=100),
    'Gender': np.random.choice(['Male', 'Female'], size=100),
    'CD4_Count': np.random.randint(200, 1500, size=100),
    'Tuberculosis_Diagnosis': np.random.choice([0, 1], size=100)
}
df = pd.DataFrame(data)

# Step 2: Preprocess the Data
df['HIV_Status'] = df['HIV_Status'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Define features and target
X = df[['Age', 'CD4_Count']]
y = df['Tuberculosis_Diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train a Random Forest Classifier
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Optional: Parameter Tuning with Grid Search
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# Re-train the model with the best parameters found by Grid Search
best_model.fit(X_train_scaled, y_train)

# Step 4: Evaluate the Model
# Make predictions
predictions = best_model.predict(X_test_scaled)

# Evaluate the model with additional metrics
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
if np.sum(y_test) > 0:  # Ensure there are positive cases
    print("ROC AUC Score:", roc_auc_score(y_test, predictions))
else:
    print("Cannot calculate ROC AUC Score: No positive cases.")
