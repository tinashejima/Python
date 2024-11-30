import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming your dataset now includes more features
data = {
    'HIV_Status': np.random.choice(['Yes', 'No'], size=100),
    'Age': np.random.randint(20, 60, size=100),
    'Gender': np.random.choice(['Male', 'Female'], size=100),
    'Income_Level': np.random.choice(['Low', 'Medium', 'High'], size=100),
    'Education_Level': np.random.choice(['Primary', 'Secondary', 'Higher'], size=100),
    'Duration_of_HIV_Infection': np.random.randint(1, 10, size=100),
    'Treatment_Adherence': np.random.choice(['Poor', 'Fair', 'Good'], size=100),
    'Tuberculosis_Diagnosis': np.random.choice([0, 1], size=100)
}
df = pd.DataFrame(data)

# Preprocessing
df['HIV_Status'] = df['HIV_Status'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Income_Level'] = df['Income_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
df['Education_Level'] = df['Education_Level'].map({'Primary': 1, 'Secondary': 2, 'Higher': 3})
df['Treatment_Adherence'] = df['Treatment_Adherence'].map({'Poor': 1, 'Fair': 2, 'Good': 3})

# Feature and Target Definition
X = df[['Age', 'Gender', 'Income_Level', 'Education_Level', 'Duration_of_HIV_Infection', 'Treatment_Adherence']]
y = df['Tuberculosis_Diagnosis']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the models
dt_model = DecisionTreeClassifier(random_state=42)
svm_model = SVC(kernel='linear', C=1, random_state=42)

# Fit the models
dt_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_scaled, y_train)

# Making predictions
dt_predictions = dt_model.predict(X_test_scaled)
svm_predictions = svm_model.predict(X_test_scaled)

# Evaluating the models
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))
