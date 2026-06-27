# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.impute import SimpleImputer
# from imblearn.over_sampling import SMOTE
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Use a non-GUI backend for Matplotlib to prevent GTK errors
# import matplotlib
# matplotlib.use('TkAgg')  # Allows figures to be displayed before saving

# # Load the dataset
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     df.drop(columns=["id"], inplace=True)  # Remove ID column as it is not useful for prediction
    
#     # Handle missing values in BMI by filling with mean value
#     imputer = SimpleImputer(strategy="mean")
#     df["bmi"] = imputer.fit_transform(df[["bmi"]])
    
#     # Encode categorical variables into numerical values
#     categorical_columns = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
#     label_encoders = {}
#     for col in categorical_columns:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#         label_encoders[col] = le
    
#     return df

# # Preprocess the dataset
# def preprocess_data(df):
#     X = df.drop(columns=["stroke"])  # Features
#     y = df["stroke"]  # Target variable
    
#     # Handle class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
#     smote = SMOTE()
#     #print(df)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     #print(X_resampled)
#     #print(y_resampled)
    
#     # Scale numerical features for better model performance
#     scaler = StandardScaler()
#     X_resampled = scaler.fit_transform(X_resampled)
    
#     # Split data into 80% training and 20% testing
#     return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # Train and evaluate multiple models
# def train_and_evaluate(X_train, X_test, y_train, y_test):
#     models = {
#         "Logistic Regression": LogisticRegression(),
#         "Random Forest": RandomForestClassifier(),
#         "SVM": SVC(),
#         "KNN": KNeighborsClassifier()
#     }
    
#     results = {}  # Dictionary to store model evaluation metrics
#     all_predictions = {}  # Dictionary to store predicted vs expected values
    
#     for name, model in models.items():
#         model.fit(X_train, y_train)  # Train the model
#         y_pred = model.predict(X_test)  # Make predictions
        
#         # Calculate evaluation metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
        
#         results[name] = {
#             "Accuracy": accuracy,
#             "Precision": precision,
#             "Recall": recall,
#             "F1 Score": f1
#         }
        
#         all_predictions[name] = (y_test, y_pred)
#         print(f"{name} Results:")
#         print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
#     return results, all_predictions

# # Load and preprocess dataset
# filepath = "healthcare-dataset-stroke-data.csv"
# df = load_data(filepath)
# X_train, X_test, y_train, y_test = preprocess_data(df)

# # Train and compare models
# results, all_predictions = train_and_evaluate(X_train, X_test, y_train, y_test)

# # Display model performance comparison plot
# plt.figure(figsize=(12, 6))
# metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
# results_df = pd.DataFrame(results).T
# results_df.plot(kind='bar', figsize=(12, 6), colormap='viridis', edgecolor='black')
# plt.xlabel("Model")
# plt.ylabel("Score")
# plt.title("Comparison of ML Models for Stroke Prediction")
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# # plt.show()
# plt.savefig("model_comparison.png")
# plt.close()

# # Display and save individual model predictions
# for model_name, (y_test, predictions) in all_predictions.items():
#     plt.figure(figsize=(6, 4))
#     sns.countplot(x=predictions, palette='coolwarm')
#     plt.xlabel("Predicted Outcome")
#     plt.ylabel("Count")
#     plt.title(f"Prediction Distribution - {model_name}")
#     plt.xticks([0, 1], ["No Stroke", "Stroke"])
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     # plt.show()
#     plt.savefig(f"prediction_{model_name.replace(' ', '_')}.png")
#     plt.close()
    
#     # Display and save Expected vs Predicted plot
#     plt.figure(figsize=(6, 4))
#     df_results = pd.DataFrame({"Expected": y_test, "Predicted": predictions})
#     sns.histplot(df_results, bins=3, kde=False, palette='coolwarm', multiple='dodge')
#     plt.xlabel("Class")
#     plt.ylabel("Count")
#     plt.title(f"Expected vs Predicted - {model_name}")
#     plt.xticks([0, 1], ["No Stroke", "Stroke"])
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     # plt.show()
#     plt.savefig(f"expected_vs_predicted_{model_name.replace(' ', '_')}.png")
#     plt.close()

# # Correlation between blood sugar, blood pressure, and stroke
# plt.figure(figsize=(8, 6))
# sns.heatmap(df[['avg_glucose_level', 'hypertension', 'stroke']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title("Correlation between Blood Sugar, Blood Pressure, and Stroke")
# # plt.show()
# plt.savefig("correlation_blood_sugar_bp_stroke.png")
# plt.close()

# print("Model comparison and predictions saved as images.")

#extra no need
# import pandas as pd
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
#     print("After Cross Validation")
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
