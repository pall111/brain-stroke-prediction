import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.drop(columns=["id"], inplace=True)  # Remove ID column as it is not useful for prediction
    
    # Handle missing values in BMI by filling with mean value
    imputer = SimpleImputer(strategy="mean")
    df["bmi"] = imputer.fit_transform(df[["bmi"]])
    
    # Encode categorical variables into numerical values
    categorical_columns = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df

# Preprocess the dataset
def preprocess_data(df):
    X = df.drop(columns=["stroke"])  # Features
    y = df["stroke"]  # Target variable
    
    # Handle class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Scale numerical features for better model performance
    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(X_resampled)
    
    # Split data into 80% training and 20% testing
    return X_resampled, y_resampled, train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train and evaluate multiple models (Before Cross-Validation)
def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n")
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }
    
    results = {}  # Dictionary to store metrics
    
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train model
        y_pred = model.predict(X_test)  # Make predictions
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = tp/ (tp +fn)  
        specificity = tn / (tn + fp)  

        results[name] = {
            "Accuracy": accuracy,
            "Sensitivity (Recall)": sensitivity,
            "Specificity": specificity
        }
        
        print(f"{name} - Accuracy : {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    
    return results

# Cross-validation function (After Cross-Validation)
def cross_validate_models(X, y, k=10):
    print("\nAfter Cross Validation")
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }
    
    cv_results = {}  # Dictionary to store mean metrics
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)  # 5-Fold Stratified Cross-Validation
    
    for name, model in models.items():
        accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        sensitivity_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
        
        specificity_scores = []  # Specificity is not directly supported, so we compute it manually
        for train_idx, test_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred).ravel()
            specificity_scores.append(tn / (tn + fp))  # Compute specificity per fold

        # Store mean and standard deviation for each metric
        mean_accuracy = accuracy_scores.mean()
        mean_sensitivity = sensitivity_scores.mean()
        mean_specificity = sum(specificity_scores) / len(specificity_scores)

        cv_results[name] = {
            "Accuracy": mean_accuracy,
            "Sensitivity (Recall)": mean_sensitivity,
            "Specificity": mean_specificity
        }
        
        print(f"{name} - Mean Accuracy : {mean_accuracy:.4f}, Sensitivity: {mean_sensitivity:.4f}, Specificity: {mean_specificity:.4f}")
    
    return cv_results

# Load and preprocess dataset
filepath = "healthcare-dataset-stroke-data.csv"
df = load_data(filepath)
X_resampled, y_resampled, (X_train, X_test, y_train, y_test) = preprocess_data(df)

# Step 1: Metrics before cross-validation
before_cv_results = train_and_evaluate(X_train, X_test, y_train, y_test)

# Step 2: Metrics after 5-Fold Cross-Validation
after_cv_results = cross_validate_models(X_resampled, y_resampled)

# Convert results to DataFrame for better visualization
cv_comparison_df = pd.DataFrame({
    "Before Cross-Validation": {model: before_cv_results[model]["Accuracy"] for model in before_cv_results},
    "After Cross-Validation": {model: after_cv_results[model]["Accuracy"] for model in after_cv_results}
})

# Sensitivity and Specificity Comparison
sensitivity_comparison_df = pd.DataFrame({
    "Before Cross-Validation": {model: before_cv_results[model]["Sensitivity (Recall)"] for model in before_cv_results},
    "After Cross-Validation": {model: after_cv_results[model]["Sensitivity (Recall)"] for model in after_cv_results}
})

specificity_comparison_df = pd.DataFrame({
    "Before Cross-Validation": {model: before_cv_results[model]["Specificity"] for model in before_cv_results},
    "After Cross-Validation": {model: after_cv_results[model]["Specificity"] for model in after_cv_results}
})

# Plot Accuracy Comparison
cv_comparison_df.plot(kind='bar', figsize=(10, 5), color=["red", "green"], edgecolor="black")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy (Before vs. After Cross-Validation)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(["Before Cross-Validation", "After Cross-Validation"])
plt.savefig("accuracy_comparison.png")
plt.close()

# Plot Sensitivity Comparison
sensitivity_comparison_df.plot(kind='bar', figsize=(10, 5), color=["blue", "orange"], edgecolor="black")
plt.xlabel("Model")
plt.ylabel("Sensitivity (Recall)")
plt.title("Model Sensitivity (Before vs. After Cross-Validation)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(["Before Cross-Validation", "After Cross-Validation"])
plt.savefig("sensitivity_comparison.png")
plt.close()

# Plot Specificity Comparison
specificity_comparison_df.plot(kind='bar', figsize=(10, 5), color=["purple", "cyan"], edgecolor="black")
plt.xlabel("Model")
plt.ylabel("Specificity")
plt.title("Model Specificity (Before vs. After Cross-Validation)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(["Before Cross-Validation", "After Cross-Validation"])
plt.savefig("specificity_comparison.png")
plt.close()

print("Plots saved: Accuracy, Sensitivity, Specificity.")

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Stroke", "Stroke"]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_confusion_artix.png")
    plt.show()
# Train Random Forest separately for confusion matrix heatmap
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Plot confusion matrix heatmap
plot_confusion_matrix_heatmap(y_test, rf_pred, model_name="Random Forest")


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.impute import SimpleImputer
# from imblearn.over_sampling import SMOTE
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# # Load the dataset
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     df.drop(columns=["id"], inplace=True)  # Remove ID column as it is not useful for prediction
    
#     # Handle missing values in BMI by filling with mean value
#     imputer = SimpleImputer(strategy="mean")
#     df["bmi"] = imputer.fit_transform(df[["bmi"]])
    
#     # Encode categorical variables into numerical values
#     categorical_columns = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
#     for col in categorical_columns:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
    
#     return df

# # Preprocess the dataset
# def preprocess_data(df):
#     X = df.drop(columns=["stroke"])  # Features
#     y = df["stroke"]  # Target variable
    
#     # Handle class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
#     smote = SMOTE()
#     X_resampled, y_resampled = smote.fit_resample(X, y)
    
#     # Scale numerical features for better model performance
#     scaler = StandardScaler()
#     X_resampled = scaler.fit_transform(X_resampled)
    
#     # Split data into 80% training and 20% testing
#     return X_resampled, y_resampled, train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # Train and evaluate multiple models (Before Cross-Validation)
# def train_and_evaluate(X_train, X_test, y_train, y_test):
#     print("\n")
#     models = {
#         "Logistic Regression": LogisticRegression(),
#         "Random Forest": RandomForestClassifier(),
#         "SVM": SVC(),
#         "KNN": KNeighborsClassifier()
#     }
    
#     results = {}  # Dictionary to store metrics
    
#     for name, model in models.items():
#         model.fit(X_train, y_train)  # Train model
#         y_pred = model.predict(X_test)  # Make predictions
        
#         # Compute confusion matrix
#         tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
#         # Compute evaluation metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         sensitivity = tp/ (tp +fn)  
#         specificity = tn / (tn + fp)  

#         results[name] = {
#             "Accuracy": accuracy,
#             "Sensitivity (Recall)": sensitivity,
#             "Specificity": specificity
#         }
        
#         print(f"{name} - Accuracy : {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    
#     return results

# # Cross-validation function (After Cross-Validation)
# def cross_validate_models(X, y, k=10):
#     print("\nAfter Cross Validation")
#     models = {
#         "Logistic Regression": LogisticRegression(),
#         "Random Forest": RandomForestClassifier(),
#         "SVM": SVC(),
#         "KNN": KNeighborsClassifier()
#     }
    
#     cv_results = {}  # Dictionary to store mean metrics
    
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)  # 5-Fold Stratified Cross-Validation
    
#     for name, model in models.items():
#         accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
#         sensitivity_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
        
#         specificity_scores = []  # Specificity is not directly supported, so we compute it manually
#         for train_idx, test_idx in skf.split(X, y):
#             model.fit(X[train_idx], y[train_idx])
#             y_pred = model.predict(X[test_idx])
#             tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred).ravel()
#             specificity_scores.append(tn / (tn + fp))  # Compute specificity per fold

#         # Store mean and standard deviation for each metric
#         mean_accuracy = accuracy_scores.mean()
#         mean_sensitivity = sensitivity_scores.mean()
#         mean_specificity = sum(specificity_scores) / len(specificity_scores)

#         cv_results[name] = {
#             "Accuracy": mean_accuracy,
#             "Sensitivity (Recall)": mean_sensitivity,
#             "Specificity": mean_specificity
#         }
        
#         print(f"{name} - Mean Accuracy : {mean_accuracy:.4f}, Sensitivity: {mean_sensitivity:.4f}, Specificity: {mean_specificity:.4f}")
    
#     return cv_results

# def predict_stroke(model, scaler):
#     print("\nEnter patient details to predict stroke risk:")
    
#     # User inputs for features
#     gender = int(input("Gender (0: Female, 1: Male): "))
#     age = float(input("Age: "))
#     hypertension = int(input("Hypertension (0: No, 1: Yes): "))
#     heart_disease = int(input("Heart Disease (0: No, 1: Yes): "))
#     ever_married = int(input("Ever Married (0: No, 1: Yes): "))
#     work_type = int(input("Work Type (0-4): "))
#     residence_type = int(input("Residence Type (0: Rural, 1: Urban): "))
#     avg_glucose_level = float(input("Average Glucose Level: "))
#     bmi = float(input("BMI: "))
#     smoking_status = int(input("Smoking Status (0-3): "))

#     # Create input array
#     input_data = np.array([[gender, age, hypertension, heart_disease, ever_married,
#                              work_type, residence_type, avg_glucose_level, bmi, smoking_status]])
    
#     # Scale input
#     input_data_scaled = scaler.transform(input_data)
    
#     # Prediction
#     prediction = model.predict(input_data_scaled)
    
#     if prediction[0] == 1:
#         print("\nPrediction: HIGH RISK of Stroke 🚨")
#     else:
#         print("\nPrediction: LOW RISK of Stroke ✅")

# # Load and preprocess dataset
# filepath = "healthcare-dataset-stroke-data.csv"
# df = load_data(filepath)
# X_resampled, y_resampled, (X_train, X_test, y_train, y_test) = preprocess_data(df)

# # Step 1: Metrics before cross-validation
# before_cv_results = train_and_evaluate(X_train, X_test, y_train, y_test)

# # Step 2: Metrics after 5-Fold Cross-Validation
# after_cv_results = cross_validate_models(X_resampled, y_resampled)

# # Convert results to DataFrame for better visualization
# cv_comparison_df = pd.DataFrame({
#     "Before Cross-Validation": {model: before_cv_results[model]["Accuracy"] for model in before_cv_results},
#     "After Cross-Validation": {model: after_cv_results[model]["Accuracy"] for model in after_cv_results}
# })

# # Sensitivity and Specificity Comparison
# sensitivity_comparison_df = pd.DataFrame({
#     "Before Cross-Validation": {model: before_cv_results[model]["Sensitivity (Recall)"] for model in before_cv_results},
#     "After Cross-Validation": {model: after_cv_results[model]["Sensitivity (Recall)"] for model in after_cv_results}
# })

# specificity_comparison_df = pd.DataFrame({
#     "Before Cross-Validation": {model: before_cv_results[model]["Specificity"] for model in before_cv_results},
#     "After Cross-Validation": {model: after_cv_results[model]["Specificity"] for model in after_cv_results}
# })

# # Plot Accuracy Comparison
# cv_comparison_df.plot(kind='bar', figsize=(10, 5), color=["red", "green"], edgecolor="black")
# plt.xlabel("Model")
# plt.ylabel("Accuracy")
# plt.title("Model Accuracy (Before vs. After Cross-Validation)")
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.legend(["Before Cross-Validation", "After Cross-Validation"])
# plt.savefig("accuracy_comparison.png")
# plt.close()

# # Plot Sensitivity Comparison
# sensitivity_comparison_df.plot(kind='bar', figsize=(10, 5), color=["blue", "orange"], edgecolor="black")
# plt.xlabel("Model")
# plt.ylabel("Sensitivity (Recall)")
# plt.title("Model Sensitivity (Before vs. After Cross-Validation)")
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.legend(["Before Cross-Validation", "After Cross-Validation"])
# plt.savefig("sensitivity_comparison.png")
# plt.close()

# # Plot Specificity Comparison
# specificity_comparison_df.plot(kind='bar', figsize=(10, 5), color=["purple", "cyan"], edgecolor="black")
# plt.xlabel("Model")
# plt.ylabel("Specificity")
# plt.title("Model Specificity (Before vs. After Cross-Validation)")
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.legend(["Before Cross-Validation", "After Cross-Validation"])
# plt.savefig("specificity_comparison.png")
# plt.close()

# print("Plots saved: Accuracy, Sensitivity, Specificity.")

# best_model = RandomForestClassifier()  # Change to your best-performing model
# best_model.fit(X_train, y_train)
# scaler = StandardScaler().fit(X_train)
# predict_stroke(best_model, scaler)