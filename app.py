"""
app.py
------
Flask web application for the AI-Based Mental Health Monitoring Solution.

Users fill in a short lifestyle/wellbeing form. The trained RandomForest
model predicts a "risk level" (Low / Moderate / High) and the app shows
a friendly result along with general self-care suggestions.

IMPORTANT: This is a student/academic project, not a medical device.
It does NOT diagnose any mental health condition. See README.md.

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

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

model = None
scaler = None
encoder = None


def load_artifacts():
    global model, scaler, encoder
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))


SUGGESTIONS = {
    "Low": [
        "Your current habits look supportive of your wellbeing — keep it up.",
        "Maintain your sleep routine and regular physical activity.",
        "Keep nurturing your social connections.",
    ],
    "Moderate": [
        "Consider small, consistent changes: an earlier bedtime or a short daily walk.",
        "Try to reduce screen time before bed to improve sleep quality.",
        "Talking to a friend, family member, or counselor can help you manage stress.",
    ],
    "High": [
        "Your responses suggest elevated stress — please consider talking to a "
        "counselor, therapist, or trusted person soon.",
        "Prioritize sleep, gentle movement, and reaching out for support.",
        "If you are in crisis or having thoughts of self-harm, please contact a "
        "local emergency number or a mental health helpline right away.",
    ],
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=FEATURE_COLUMNS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form[col]) for col in FEATURE_COLUMNS]
    except (KeyError, ValueError):
        return render_template(
            "index.html",
            features=FEATURE_COLUMNS,
            error="Please fill in all fields with valid numbers.",
        )

    X = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(X)
    pred_encoded = model.predict(X_scaled)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba(X_scaled)[0]
    prob_dict = {
        cls: round(float(p) * 100, 1) for cls, p in zip(encoder.classes_, probs)
    }

    return render_template(
        "result.html",
        risk_level=pred_label,
        probabilities=prob_dict,
        suggestions=SUGGESTIONS.get(pred_label, []),
    )


load_artifacts()

if __name__ == "__main__":
    app.run(debug=True)
