import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_excel(
    "/rds/general/user/ka325/home/synthseg/SynthSeg/ML_dataset.xlsx"
)

print(f"\nLoaded {len(df)} subjects")

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

X = df[
    [
        "age at visit",
        "MMSE",
        "EC_volume",
        "EC_percentage"
    ]
]

# --------------------------------------------------
# LABEL
# --------------------------------------------------

y = df["Label"]

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

model.fit(X_train, y_train)

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

classifier = model.named_steps["classifier"]

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

for feature, coef in zip(
    X.columns,
    classifier.coef_[0]
):
    print(f"{feature}: {coef:.4f}")

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    "/rds/general/user/ka325/home/synthseg/SynthSeg/ad_classifier.joblib"
)

print("\n✅ Model saved as:")
print("ad_classifier.joblib")