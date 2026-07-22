"""
generate_dataset.py
====================
Generates a realistic synthetic Employee Salary dataset (~10,000 rows)
modeled after the UCI Adult Census Income dataset.

Columns produced:
    age, workclass, education, education_num, marital_status, occupation,
    relationship, race, gender, hours_per_week, experience, native_country, salary

Target: salary (<=50K / >50K)

Usage:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SAMPLES = 10000
OUTPUT_FILE = "employee_salary_data.csv"

np.random.seed(RANDOM_STATE)


def _build_education_mapping():
    """Return ordered education levels with numeric codes."""
    return {
        "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
        "9th": 5, "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9,
        "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
        "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
    }


def _weighted_choice(options, weights, size):
    """Sample from *options* according to *weights*."""
    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    return np.random.choice(options, size=size, p=probs)


def generate_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    """
    Generate a synthetic employee salary dataset.

    Parameters
    ----------
    n : int
        Number of rows to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with all feature columns and the target ``salary``.
    """
    print(f"[INFO] Generating {n} synthetic employee records …")

    edu_map = _build_education_mapping()
    edu_labels = list(edu_map.keys())
    edu_weights = [1, 1, 1, 2, 2, 3, 3, 3, 15, 12, 5, 4, 18, 8, 3, 2]

    # ── Core features ────────────────────────────────────────
    age = np.random.randint(17, 72, size=n)

    education = _weighted_choice(edu_labels, edu_weights, n)
    education_num = np.array([edu_map[e] for e in education])

    workclass_options = [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov", "Without-pay",
    ]
    workclass_weights = [65, 8, 4, 5, 8, 6, 1]
    workclass = _weighted_choice(workclass_options, workclass_weights, n)

    occupation_options = [
        "Tech-support", "Craft-repair", "Sales", "Exec-managerial",
        "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
        "Adm-clerical", "Farming-fishing", "Transport-moving",
        "Protective-serv", "Other-service", "Priv-house-serv",
    ]
    occupation_weights = [5, 10, 10, 12, 14, 6, 6, 10, 3, 5, 3, 12, 2]
    occupation = _weighted_choice(occupation_options, occupation_weights, n)

    marital_options = [
        "Married-civ-spouse", "Never-married", "Divorced",
        "Separated", "Widowed", "Married-spouse-absent",
    ]
    marital_weights = [40, 30, 14, 5, 6, 5]
    marital_status = _weighted_choice(marital_options, marital_weights, n)

    relationship_options = [
        "Husband", "Not-in-family", "Own-child",
        "Unmarried", "Wife", "Other-relative",
    ]
    relationship_weights = [35, 25, 15, 10, 8, 7]
    relationship = _weighted_choice(relationship_options, relationship_weights, n)

    race_options = ["White", "Black", "Asian-Pac-Islander",
                    "Amer-Indian-Eskimo", "Other"]
    race_weights = [75, 12, 6, 3, 4]
    race = _weighted_choice(race_options, race_weights, n)

    gender = _weighted_choice(["Male", "Female"], [60, 40], n)

    hours_per_week = np.clip(
        np.random.normal(40, 10, size=n).astype(int), 1, 80
    )

    # Experience = age − years_of_education − ~6  (clamped ≥ 0)
    experience = np.clip(age - education_num - 6 + np.random.randint(-2, 3, n), 0, 50)

    country_options = [
        "United-States", "Mexico", "Philippines", "Germany",
        "India", "Canada", "Puerto-Rico", "El-Salvador",
        "Cuba", "England", "China", "South", "Jamaica",
        "Italy", "Dominican-Republic", "Japan", "Other",
    ]
    country_weights = [70, 3, 2, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9]
    native_country = _weighted_choice(country_options, country_weights, n)

    # ── Target: salary ───────────────────────────────────────
    # Create a realistic probability based on correlated features
    prob_high = (
        0.05
        + 0.012 * (age - 17)
        + 0.03 * (education_num - 1)
        + 0.008 * experience
        + 0.004 * (hours_per_week - 20)
    )
    # Occupation boost
    high_occ = {"Exec-managerial", "Prof-specialty", "Tech-support", "Protective-serv"}
    prob_high += np.array([0.12 if o in high_occ else 0.0 for o in occupation])
    # Workclass boost
    prob_high += np.array([0.08 if w == "Self-emp-inc" else 0.0 for w in workclass])

    prob_high = np.clip(prob_high, 0.03, 0.95)
    salary_binary = np.random.binomial(1, prob_high)
    salary = np.where(salary_binary == 1, ">50K", "<=50K")

    # ── Inject ~2 % missing values in select columns ─────────
    def _inject_missing(arr, frac=0.02):
        idx = np.random.choice(n, size=int(n * frac), replace=False)
        arr = arr.astype(object)
        arr[idx] = np.nan
        return arr

    workclass = _inject_missing(workclass)
    occupation = _inject_missing(occupation)
    native_country = _inject_missing(native_country, frac=0.01)

    # ── Assemble DataFrame ───────────────────────────────────
    df = pd.DataFrame({
        "age": age,
        "workclass": workclass,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "hours_per_week": hours_per_week,
        "experience": experience,
        "native_country": native_country,
        "salary": salary,
    })

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Target distribution:\n{df['salary'].value_counts()}")
    return df


def save_dataset(df: pd.DataFrame, path: str = OUTPUT_FILE):
    """Save the dataset to a CSV file."""
    df.to_csv(path, index=False)
    print(f"[INFO] Dataset saved to '{path}'")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = generate_dataset()
    save_dataset(dataset)
    print("[DONE] Dataset generation complete.")
