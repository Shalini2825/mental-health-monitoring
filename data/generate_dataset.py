"""
generate_dataset.py
--------------------
Generates a synthetic lifestyle/wellbeing survey dataset used to train the
AI-Based Mental Health Monitoring Solution.

NOTE ON DATA: Real, ethically-sourced mental health datasets (e.g. clinical
survey data) usually require a data-use agreement and cannot be bundled in a
public GitHub repo. For this academic project we generate a synthetic but
statistically realistic dataset from simple, well-documented rules so the
project is fully reproducible offline. Swap this file for a real dataset
loader if you have IRB-approved / licensed data.

Run:
    python data/generate_dataset.py
Produces:
    data/survey_data.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 2000

np.random.seed(RANDOM_SEED)


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    age = np.random.randint(16, 60, n_samples)
    sleep_hours = np.clip(np.random.normal(6.5, 1.5, n_samples), 2, 10)
    screen_time_hours = np.clip(np.random.normal(6, 2.5, n_samples), 0, 16)
    physical_activity_hours = np.clip(np.random.exponential(1.0, n_samples), 0, 7)
    social_interaction_score = np.clip(np.random.normal(5, 2, n_samples), 0, 10)
    work_study_hours = np.clip(np.random.normal(7, 2, n_samples), 0, 16)
    self_rated_mood = np.clip(np.random.normal(6, 2, n_samples), 0, 10)
    academic_work_pressure = np.clip(np.random.normal(5.5, 2, n_samples), 0, 10)
    financial_stress = np.clip(np.random.normal(4, 2.5, n_samples), 0, 10)
    support_system_score = np.clip(np.random.normal(6, 2, n_samples), 0, 10)

    # A simple, transparent scoring rule combines the factors into a
    # continuous "stress index". This is a heuristic used only to create
    # plausible labels for a learning project -- it is NOT a validated
    # clinical instrument.
    stress_index = (
        -0.9 * (sleep_hours - 6.5)
        + 0.35 * (screen_time_hours - 6)
        - 0.5 * physical_activity_hours
        - 0.4 * (social_interaction_score - 5)
        + 0.3 * (work_study_hours - 7)
        - 0.6 * (self_rated_mood - 6)
        + 0.55 * (academic_work_pressure - 5.5)
        + 0.45 * (financial_stress - 4)
        - 0.5 * (support_system_score - 6)
        + np.random.normal(0, 1.5, n_samples)  # noise
    )

    risk_level = pd.cut(
        stress_index,
        bins=[-np.inf, -1.0, 1.5, np.inf],
        labels=["Low", "Moderate", "High"],
    )

    df = pd.DataFrame(
        {
            "age": age,
            "sleep_hours": np.round(sleep_hours, 1),
            "screen_time_hours": np.round(screen_time_hours, 1),
            "physical_activity_hours": np.round(physical_activity_hours, 1),
            "social_interaction_score": np.round(social_interaction_score, 1),
            "work_study_hours": np.round(work_study_hours, 1),
            "self_rated_mood": np.round(self_rated_mood, 1),
            "academic_work_pressure": np.round(academic_work_pressure, 1),
            "financial_stress": np.round(financial_stress, 1),
            "support_system_score": np.round(support_system_score, 1),
            "risk_level": risk_level.astype(str),
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/survey_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["risk_level"].value_counts())
