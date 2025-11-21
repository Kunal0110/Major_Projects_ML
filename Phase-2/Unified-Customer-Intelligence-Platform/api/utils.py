import pandas as pd

def dict_to_dataframe(d: dict) -> pd.DataFrame:
    # Convert customer dict to single-row DataFrame
    return pd.DataFrame([d])