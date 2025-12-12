import pandas as pd
from pathlib import Path
import shap 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from model_utils.preprocessing import build_preprocessor
from model_utils.metrics import classification_metrics
from model_utils.io import save_model

#Paths
Feature_Dir = Path("data/features")
Gold_Path = Path("data/gold/customer_gold_master.parquet")
Out_Dir = Path("models/churn")
Out_Dir.mkdir(parents=True, exist_ok=True)

def load_gold_table():
    return pd.read_parquet(Gold_Path)

def train_churn():
    df = load_gold_table()
    print(df.head())

    # Train/val split

    target = "churn"
    X = df.drop(columns=["churn", "customer_id"])
    y = df[target]

    # Identify feature types

    numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.to_list()
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.to_list()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    #Preprocessor
    pre = build_preprocessor(numerical_cols, categorical_cols)

    #Baseline logistic regression

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg_pipe = Pipeline([("preprocessor", pre), ("model", logreg)])
    logreg_pipe.fit(X_train, y_train)

    logreg_pred = logreg_pipe.predict(X_test)
    logreg_proba = logreg_pipe.predict_proba(X_test)[:,1]

    metrics_logreg = classification_metrics(y_test, logreg_pred, logreg_proba)
    print("Logistic Regression:", metrics_logreg)

    save_model(logreg_pipe, Out_Dir/"baseline_logreg.pkl")

    # XGBoost with RandomizedSearchCV

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        use_label_encoder=False
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Preprocess data first
    X_train_processed = pre.fit_transform(X_train)
    X_test_processed = pre.transform(X_test)
    
    # Feature selection on processed data
    selector = SelectKBest(f_classif, k=20)
    X_train_selected = selector.fit_transform(X_train_processed, y_train)
    X_test_selected = selector.transform(X_test_processed)
    
    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

    param_dist = {
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "n_estimators": [100, 200, 300]
    }

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=30,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train_balanced, y_train_balanced)
    best_xgb = search.best_estimator_

    xgb_pred = best_xgb.predict(X_test_selected)
    xgb_proba = best_xgb.predict_proba(X_test_selected)[:,1]

    metrics_xgb = classification_metrics(y_test, xgb_pred, xgb_proba)
    print("XGB Best:", metrics_xgb)

    save_model(best_xgb, Out_Dir/"xgb_best.pkl")
    save_model(pre, Out_Dir/"preprocessor.pkl")
    save_model(selector, Out_Dir/"feature_selector.pkl")

    # Stacking Ensemble

    estimators = [
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=300))
    ]

    final_est = LogisticRegression(max_iter=1000)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        stack_method="predict_proba",
        n_jobs=-1
    )

    pipe_stack = Pipeline([
        ("preprocessor", pre),
        ("model", stack)
    ])

    pipe_stack.fit(X_train, y_train)

    stack_pred = pipe_stack.predict(X_test)
    stack_proba = pipe_stack.predict_proba(X_test)[:,1]

    metrics_stack = classification_metrics(y_test, stack_pred, stack_proba)
    print("Stacked Ensemble:", metrics_stack)

    save_model(pipe_stack, Out_Dir/"stacking_model.pkl")

    # Shap Explainability
    print("Generating SHAP values for XGB --")

    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test_selected)

    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test_selected, show=False)
    plt.savefig(Out_Dir/"shap_summary.png")
    plt.close()

    print("SHAP plot saved.")

if __name__ == "__main__":
    train_churn()
