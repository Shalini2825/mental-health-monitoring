"""
train_model.py
---------------
Trains a RandomForestClassifier to predict a wellbeing "risk_level"
(Low / Moderate / High) from lifestyle survey features, and saves the
trained model + preprocessing objects for use by the Flask app.

Run:
    python model/train_model.py
Produces (in model/):
    model.pkl, scaler.pkl, label_encoder.pkl, feature_importance.png,
    confusion_matrix.png, metrics.txt
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_SEED = 42
DATA_PATH = os.path.join("data", "survey_data.csv")
MODEL_DIR = "model"

FEATURE_COLUMNS = [
    "age",
    "sleep_hours",
    "screen_time_hours",
    "physical_activity_hours",
    "social_interaction_score",
    "work_study_hours",
    "self_rated_mood",
    "academic_work_pressure",
    "financial_stress",
    "support_system_score",
]
TARGET_COLUMN = "risk_level"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_, digits=3
    )
    print(report)

    accuracy = (y_pred == y_test).mean()
    print(f"Test accuracy: {accuracy:.3f}")

    # Save metrics to file
    with open(os.path.join(MODEL_DIR, "metrics.txt"), "w") as f:
        f.write(f"Test accuracy: {accuracy:.3f}\n\n")
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance plot
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=True)
    plt.figure(figsize=(6, 5))
    importances.plot(kind="barh", color="#4C72B0")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    # Save model artifacts
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"Saved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
