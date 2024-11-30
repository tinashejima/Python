import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to scale new data
def scale_data(data):
    # Assuming the scaler was fitted on the original dataset which included 'Income_Level' and 'Education_Level'
    # You need to ensure that the scaler used here matches exactly what was used during training.
    # This might require saving the scaler state and loading it here, or re-fitting it on the combined training data.
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Load the Decision Tree model
with open('best_decision_tree_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)

# Load the SVM model
with open('best_svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Example input data
new_data = np.array([[30, 500, 2, 3]])  # Updated to include 'Income_Level' and 'Education_Level'

# Scale the new data
scaled_new_data = scale_data(new_data)

# Make predictions with both models
dt_prediction = dt_model.predict(scaled_new_data)
svm_prediction = svm_model.predict(scaled_new_data)

print("Decision Tree Prediction:", dt_prediction[0])
print("SVM Prediction:", svm_prediction[0])
