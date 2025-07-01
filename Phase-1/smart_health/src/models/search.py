"""
Grid / Random search for Random Forest & XGBoost
"""

from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import average_precision_score, make_scorer

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib, numpy as np

from src.data.clean import clean_and_save
from src.models.pipeline import make_pipeline

# scoring
def pr_auc_scorer():
    def _custom(y_true, y_pred, **_):
        if getattr(y_pred, "ndim", 0) == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]  # positive-class proba
        return average_precision_score(y_true, y_pred)

    return make_scorer(_custom, needs_proba=True)


# save paths
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# training routine 
def train():
    df = clean_and_save()
    X = df.drop(columns="readmitted_in_30_days")
    y = df["readmitted_in_30_days"]

    # class imbalance helper for XGBoost
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    searches: dict[str, RandomizedSearchCV] = {}

    #  Random Forest 
    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_grid = {
        "clf__n_estimators":  np.linspace(200, 600, 5, dtype=int),
        "clf__max_depth":     [None, 8, 12],
        "clf__min_samples_leaf": [1, 3, 5],
    }

    rf_search = RandomizedSearchCV(
        make_pipeline(rf, df),
        rf_grid,
        n_iter=30,
        scoring=pr_auc_scorer(),
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    rf_search.fit(X, y)
    searches["rf"] = rf_search

    # XGBoost
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        max_depth=6,  # sensible default
    )
    xgb_grid = {
        "clf__n_estimators":  np.linspace(300, 700, 5, dtype=int),
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample":     [0.7, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.9, 1.0],
    }

    xgb_search = RandomizedSearchCV(
        make_pipeline(xgb, df),
        xgb_grid,
        n_iter=40,
        scoring=pr_auc_scorer(),
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    xgb_search.fit(X, y)
    searches["xgb"] = xgb_search

    # choose & persist best
    best_name, best_search = max(searches.items(), key=lambda kv: kv[1].best_score_)
    best_estimator = best_search.best_estimator_
    best_auprc     = best_search.best_score_

    joblib.dump(best_estimator, MODEL_DIR / "best_model.pkl", compress=3)
    json.dump(
        {"model": best_name, "pr_auc": float(best_auprc)},
        open(MODEL_DIR / "metadata.json", "w"),
        indent=2,
    )
    print(f"{best_name} saved PR-AUC: {best_auprc:.3f}")


if __name__ == "__main__":
    train()