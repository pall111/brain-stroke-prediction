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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Global encoders and scaler
label_encoders = {}
scaler = StandardScaler()

# Load and encode data
def load_data(filepath, is_train=True):
    df = pd.read_csv(filepath)
    df.drop(columns=["id"], inplace=True, errors='ignore')
    imputer = SimpleImputer(strategy="mean")
    df["bmi"] = imputer.fit_transform(df[["bmi"]])

    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    for col in cat_cols:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col] = le.transform(df[col])
    return df

# Preprocess training data
def preprocess_training_data(df):
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
    X = scaler.fit_transform(X)
    return X, y

# Preprocess test data
def preprocess_test_data(df):
    X = scaler.transform(df)
    return X

# Plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

# Plot predicted vs actual
def plot_predicted_vs_actual(y_true, y_pred, model_name):
    true_counts = [sum(y_true == 0), sum(y_true == 1)]
    pred_counts = [sum(y_pred == 0), sum(y_pred == 1)]
    labels = ["No Stroke", "Stroke"]

    x = range(2)
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, true_counts, width, label='Actual', color='skyblue')
    ax.bar([p + width for p in x], pred_counts, width, label='Predicted', color='salmon')

    ax.set_xlabel('Outcome')
    ax.set_ylabel('Count')
    ax.set_title(f'{model_name} - Predicted vs Actual')
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{model_name}_predicted_vs_actual.png")
    plt.close()

# Train & evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }

    trained = {}
    print("\nBefore Cross-Validation:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        y_pred = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, name)
        plot_predicted_vs_actual(y_test, y_pred, name)

        print(f"{name}: Accuracy={accuracy:.4f}, Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}")
    return trained

# Cross-validation
def cross_validate(X, y):
    print("\nAfter Cross-Validation:")
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()
        sens = cross_val_score(model, X, y, cv=skf, scoring="recall").mean()
        specs = []
        for train_idx, test_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred).ravel()
            specs.append(tn / (tn + fp))
        spec = sum(specs) / len(specs)
        results.append((name, acc, sens, spec))
        print(f"{name}: Accuracy={acc:.4f}, Sensitivity={sens:.4f}, Specificity={spec:.4f}")

    df = pd.DataFrame(results, columns=["Model", "Accuracy", "Sensitivity", "Specificity"])
    df.set_index("Model")[["Accuracy", "Sensitivity", "Specificity"]].plot(kind="bar", figsize=(10, 6))
    plt.title("Model Comparison After Cross-Validation")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()

# Final prediction without stroke labels
def evaluate_on_unlabeled_test(trained_models, test_df):
    X_test = preprocess_test_data(test_df)
    model = trained_models["Random Forest"]
    predictions = model.predict(X_test)

    print("\nStroke Prediction Results:")
    for idx, pred in enumerate(predictions, 1):
        print(f"Patient {idx}: {'CAN have brain stroke' if pred == 1 else 'will NOT have brain stroke'}")

    pred_df = pd.DataFrame(predictions, columns=["Predicted Stroke"])
    pred_df["Predicted Stroke"] = pred_df["Predicted Stroke"].map({0: "No Stroke", 1: "Stroke"})
    sns.countplot(data=pred_df, x="Predicted Stroke", hue="Predicted Stroke", palette=["skyblue", "salmon"], legend=False)
    plt.title("Prediction Distribution - Random Forest")
    plt.xlabel("Predicted Outcome")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("test_prediction_distribution.png")
    plt.close()

    test_df["Predicted Stroke"] = predictions
    test_df.to_csv("stroke_predictions.csv", index=False)

# Main
def main():
    train_df = load_data("healthcare-dataset-stroke-data.csv", is_train=True)
    X, y = preprocess_training_data(train_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)
    cross_validate(X, y)

    test_path = input("\nEnter path of test dataset (CSV): ")
    test_df = load_data(test_path, is_train=False)
    evaluate_on_unlabeled_test(trained_models, test_df)

if __name__ == "__main__":
    main()