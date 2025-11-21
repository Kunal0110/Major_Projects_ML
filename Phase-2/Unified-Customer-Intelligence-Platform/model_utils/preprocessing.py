from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_preprocessor(numeric_cols, categorical_cols):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scalar", StandardScaler())
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, categorical_cols)
    ], remainder="drop")

    return pre