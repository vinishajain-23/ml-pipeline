"""Compare a low-code workflow (PyCaret) with a manual one (scikit-learn)
for predicting telecom customer churn.

Run with:  uv run discovery.py
"""

import shutil
import time

import joblib
import matplotlib

matplotlib.use("Agg")  # no pop-up windows, just save files
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "data/telco_churn.csv"
TARGET = "Churn"


def load_data():
    df = pd.read_csv(DATA_PATH)
    # TotalCharges is blank for 11 brand-new customers (tenure 0), so pandas
    # reads the column as text. Turn it into numbers; blanks become NaN and
    # get imputed later.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def save_table_png(table, path):
    """Save a DataFrame as a PNG so the comparison table ends up in images/."""
    fig, ax = plt.subplots(figsize=(12, 0.3 * len(table) + 0.5))
    ax.axis("off")
    shown = table.copy()
    for col in shown.select_dtypes("number").columns:
        shown[col] = shown[col].map("{:.4f}".format)
    t = ax.table(cellText=shown.values, colLabels=shown.columns, loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.auto_set_column_width(list(range(len(shown.columns))))
    t.scale(1, 1.4)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


df = load_data()
print(f"Loaded {len(df):,} rows and {df.shape[1] - 1} features\n")

# ---------------------------------------------------------------------------
# 1. PyCaret workflow
# ---------------------------------------------------------------------------
from pycaret.classification import compare_models, plot_model, pull, setup

start = time.perf_counter()

setup(data=df, target=TARGET, session_id=42, verbose=False)
top3 = compare_models(n_select=3, verbose=False)
comparison = pull()
best = top3[0]

plot_path = plot_model(best, plot="confusion_matrix", save=True)
shutil.move(plot_path, "images/pycaret_confusion_matrix.png")

pycaret_seconds = time.perf_counter() - start

print("PyCaret model comparison (10-fold CV on the training data):")
print(comparison.to_string())
comparison.to_csv("images/pycaret_comparison.csv")
save_table_png(comparison.reset_index(drop=True), "images/pycaret_comparison.png")
print(f"\nTop 3 models: {[type(m).__name__ for m in top3]}")
print(f"PyCaret section took {pycaret_seconds:.1f} seconds\n")

# ---------------------------------------------------------------------------
# 2. scikit-learn workflow: rebuild PyCaret's best model by hand
# ---------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

start = time.perf_counter()

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

numeric_cols = X.select_dtypes("number").columns.tolist()
categorical_cols = X.select_dtypes("object").columns.tolist()

numeric_steps = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)
categorical_steps = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocess = ColumnTransformer(
    [
        ("num", numeric_steps, numeric_cols),
        ("cat", categorical_steps, categorical_cols),
    ]
)

# PyCaret ranked Logistic Regression first. PyCaret uses max_iter=1000,
# so I do the same.
model = Pipeline(
    [
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)
model.fit(X_train, y_train)

print("scikit-learn classification report (30% held-out test set):")
print(classification_report(y_test, model.predict(X_test), digits=4))

joblib.dump(model, "model.joblib")
sklearn_seconds = time.perf_counter() - start

print(f"scikit-learn section took {sklearn_seconds:.1f} seconds")
print("Saved fitted pipeline to model.joblib")

if type(best).__name__ != "LogisticRegression":
    print(
        f"\nNOTE: PyCaret's best model on this run was {type(best).__name__}, "
        "not LogisticRegression. Update the classifier step above to match."
    )
