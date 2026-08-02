"""
Trains a Random Forest to predict whether a tech employee is likely to seek
mental health treatment, based on the OSMI 2014 Mental Health in Tech survey.
Rebuilds (in Python) the modeling approach from the original R notebooks,
and saves artifacts for the Streamlit demo app.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

RANDOM_STATE = 42

df = pd.read_csv("survey.csv")

# ---------- Clean ----------
df = df.drop(columns=["Timestamp", "comments", "state", "Country", "tech_company",
                       "seek_help", "phys_health_consequence", "phys_health_interview",
                       "mental_health_interview", "self_employed"])

# Clean Age
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df.loc[(df["Age"] < 15) | (df["Age"] > 80), "Age"] = np.nan
df["Age"] = df["Age"].fillna(df["Age"].median())

# Clean Gender into 3 buckets
def clean_gender(g):
    g = str(g).strip().lower()
    male_set = {"male", "m", "man", "cis male", "cis man", "male-ish", "maile",
                "mal", "male (cis)", "make", "guy (-ish) ^_^", "malr", "msle"}
    female_set = {"female", "f", "woman", "cis female", "cis-female/femme",
                  "femake", "female (cis)", "femail"}
    if g in male_set:
        return "Male"
    if g in female_set:
        return "Female"
    return "Other/Non-binary"

df["Gender"] = df["Gender"].apply(clean_gender)

# Fill remaining NAs
df["work_interfere"] = df["work_interfere"].fillna("Don't know")
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mode()[0])

# ---------- Encode ----------
FEATURES = [
    "Age", "Gender", "family_history", "work_interfere", "no_employees",
    "remote_work", "benefits", "care_options", "wellness_program",
    "anonymity", "leave", "mental_health_consequence", "coworkers",
    "supervisor", "mental_vs_physical", "obs_consequence",
]
TARGET = "treatment"

X = df[FEATURES].copy()
y = df[TARGET].copy()

encoders = {}
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
X = X.astype(float)

y_le = LabelEncoder()
y_enc = y_le.fit_transform(y)  # No=0, Yes=1

# ---------- Train ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
)

model = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, proba)
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {auc:.4f}")

# ---------- Save artifacts ----------
joblib.dump(model, "model.joblib")
joblib.dump(encoders, "encoders.joblib")
joblib.dump(y_le, "target_encoder.joblib")
joblib.dump(FEATURES, "features.joblib")
joblib.dump(X_test, "X_test.joblib")  # for SHAP background/reference

# Save option lists + defaults for the UI
option_lists = {}
defaults = {}
raw_df = pd.read_csv("survey.csv")
raw_df["Gender"] = raw_df["Gender"].apply(clean_gender)
for col in FEATURES:
    if col == "Age":
        defaults[col] = int(df["Age"].median())
        continue
    if col == "Gender":
        vals = sorted(raw_df["Gender"].unique().tolist())
    else:
        vals = sorted(df[col].astype(str).unique().tolist())
    option_lists[col] = vals
    defaults[col] = df[col].mode()[0] if not pd.api.types.is_numeric_dtype(df[col]) else df[col].median()

joblib.dump(option_lists, "option_lists.joblib")
joblib.dump(defaults, "defaults.joblib")
joblib.dump({"accuracy": acc, "auc": auc, "n_train": len(X_train), "n_test": len(X_test)}, "metrics.joblib")

print("Artifacts saved.")
