
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import pickle
import matplotlib.pyplot as plt

# 1. Data Collection
try:
    df = pd.read_csv("diabetes_dataset.csv")
except FileNotFoundError:
    print("Error: 'diabetes_dataset.csv' not found. Please make sure the dataset is in the correct directory.")
    exit()

# 2. Feature Engineering (and Preparation)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Development & 4. Training and Validation
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Support Vector Machine": SVC(random_state=42, probability=True)
}

results = {}

for name, model in models.items():
    print(f"--- Training and Evaluating {name} ---")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC-ROC": roc_auc
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}
")

# 5. Interpretability integration
print("--- Generating SHAP Summary Plots for Model Interpretability ---")
plt.figure()
# For Random Forest
explainer_rf = shap.TreeExplainer(models["Random Forest"])
shap_values_rf = explainer_rf.shap_values(X_test_scaled)
shap.summary_plot(shap_values_rf, X_test, plot_type="bar", show=False)
plt.title("Random Forest SHAP Summary Plot")
plt.savefig("shap_summary_random_forest.png")
plt.close()


# For Logistic Regression
explainer_lr = shap.LinearExplainer(models["Logistic Regression"], X_train_scaled)
shap_values_lr = explainer_lr.shap_values(X_test_scaled)
plt.figure()
shap.summary_plot(shap_values_lr, X_test, plot_type="bar", show=False)
plt.title("Logistic Regression SHAP Summary Plot")
plt.savefig("shap_summary_logistic_regression.png")
plt.close()


# For SVM
# Note: Using KernelExplainer for SVM which can be slow. We use a sample of the training data as the background.
explainer_svm = shap.KernelExplainer(models["Support Vector Machine"].predict_proba, shap.sample(X_train_scaled, 50))
shap_values_svm = explainer_svm.shap_values(shap.sample(X_test_scaled, 50))
plt.figure()
shap.summary_plot(shap_values_svm[1], shap.sample(X_test, 50), plot_type="bar", show=False)
plt.title("SVM SHAP Summary Plot")
plt.savefig("shap_summary_svm.png")
plt.close()

print("SHAP summary plots saved as PNG files.")

# 6. Model selection
best_model_name = max(results, key=lambda name: results[name]["AUC-ROC"])
best_model = models[best_model_name]

print("\n--- Model Selection ---")
print(f"Best performing model based on AUC-ROC: {best_model_name}")

# 7. Deployment and Evaluation (Saving the model)
with open("best_diabetes_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest model ('{best_model_name}') saved as 'best_diabetes_model.pkl'")
