# Heart Disease Prediction - Intermediate Level
# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # To save the model

# 1. Load the Dataset
# Replace 'Data.csv' with the path to your dataset
data = pd.read_csv("Data.csv")  

# Display basic information about the dataset
print("Dataset Head:")
print(data.head())
print("\nDataset Information:")
print(data.info())

# 2. Exploratory Data Analysis (EDA)
print("\nChecking for Missing Values:")
print(data.isnull().sum())

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3. Feature Engineering
# Define the target variable and features
X = data.drop("target", axis=1)  # Drop the 'target' column
y = data["target"]  # Target column: Heart Disease (1 = Yes, 0 = No)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Training and Comparison
# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)

# Random Forest Classifier
rf_clf = RandomForestClassifier()
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15]}  # Hyperparameter tuning
grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Best Random Forest Model
best_rf_clf = grid_search.best_estimator_
y_pred_rf = best_rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# 6. Model Evaluation
print("\n--- Logistic Regression Results ---")
print(f"Accuracy: {accuracy_log:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

print("\n--- Random Forest Results ---")
print(f"Accuracy: {accuracy_rf:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Compare Model Performances
models = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy_log, accuracy_rf]
})
print("\nModel Comparison:")
print(models)

# 7. Save the Best Model
if accuracy_rf > accuracy_log:
    print("\nSaving Random Forest Model...")
    joblib.dump(best_rf_clf, "heart_disease_rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
else:
    print("\nSaving Logistic Regression Model...")
    joblib.dump(log_reg, "heart_disease_log_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully!")
