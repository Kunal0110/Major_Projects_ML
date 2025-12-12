import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from pathlib import Path
from sklearn.pipeline import Pipeline
import numpy as np

from model_utils.preprocessing import build_preprocessor
from model_utils.metrics import regression_metrics
from model_utils.io import save_model

Gold_Path = Path("data/gold/customer_gold_master.parquet")
Out_Dir = Path("models/clv")
Out_Dir.mkdir(parents=True, exist_ok=True)

def train_clv():
    df = pd.read_parquet(Gold_Path)

    # Enhanced CLV calculation
    df["future_value"] = (
        df["avg_monthly_revenue"] * 12 * 
        (1 - df["churn"]) * 
        np.exp(-df["tenure_months"] / 24)
    )

    target = "future_value"
    ignore = ["churn", "customer_id"]

    X = df.drop(columns=[target] + ignore)
    y = df[target]

    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    pre = build_preprocessor(numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess data first
    X_train_processed = pre.fit_transform(X_train)
    X_test_processed = pre.transform(X_test)
    
    # Feature selection on processed data
    selector = SelectKBest(f_regression, k=25)
    X_train_selected = selector.fit_transform(X_train_processed, y_train)
    X_test_selected = selector.transform(X_test_processed)

    # Hyperparameter tuning
    param_dist = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0]
    }
    
    model = XGBRegressor(random_state=42)
    
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_train_selected, y_train)
    best_model = search.best_estimator_

    preds = best_model.predict(X_test_selected)
    metrics = regression_metrics(y_test, preds)

    print("CLV Model Metrics:", metrics)
    print("Best Parameters:", search.best_params_)

    save_model(best_model, Out_Dir / "clv_model.pkl")
    save_model(selector, Out_Dir / "clv_feature_selector.pkl")
    save_model(pre, Out_Dir / "clv_preprocessor.pkl")

if __name__ == "__main__":
    train_clv()