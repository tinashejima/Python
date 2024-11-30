import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pickle

# Step 1: Generate Fictitious Dataset with Additional Features
np.random.seed(42)
data = {
    'HIV_Status': np.random.choice(['Yes', 'No'], size=100),
    'Age': np.random.randint(20, 60, size=100),
    'Gender': np.random.choice(['Male', 'Female'], size=100),
    'Income_Level': np.random.choice(['Low', 'Medium', 'High'], size=100),  # New feature
    'Education_Level': np.random.choice(['Primary', 'Secondary', 'Higher'], size=100),  # New feature
    'CD4_Count': np.random.randint(200, 1500, size=100),
    'Tuberculosis_Diagnosis': np.random.choice([0, 1], size=100)
}
df = pd.DataFrame(data)

# Step 2: Preprocess the Data
df['HIV_Status'] = df['HIV_Status'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Income_Level'] = df['Income_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})  # Encode new feature
df['Education_Level'] = df['Education_Level'].map({'Primary': 1, 'Secondary': 2, 'Higher': 3})  # Encode new feature

# Define features and target
X = df[['Age', 'Income_Level', 'Education_Level', 'CD4_Count']]  # Updated features
y = df['Tuberculosis_Diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train Decision Tree Classifier with Hyperparameter Tuning
param_grid_dt = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20]}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train_scaled, y_train)
best_dt = grid_search_dt.best_estimator_

# Step 5: Train SVM Classifier with Hyperparameter Tuning
param_grid_svm = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.1, 1, 10]}
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train_scaled, y_train)
best_svm = grid_search_svm.best_estimator_

# Step 6: Evaluate Both Models with Additional Metrics
# Decision Tree Evaluation
dt_predictions = best_dt.predict(X_test_scaled)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_predictions))

# SVM Evaluation
svm_predictions = best_svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

# Step 7: Export the Models
# Saving the Best Decision Tree model
with open('best_decision_tree_model.pkl', 'wb') as file:
    pickle.dump(best_dt, file)

# Saving the Best SVM model
with open('best_svm_model.pkl', 'wb') as file:
    pickle.dump(best_svm, file)
