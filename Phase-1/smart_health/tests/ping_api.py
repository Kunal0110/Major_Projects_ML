import json, requests, pandas as pd, pathlib as p
root   = p.Path(__file__).resolve().parents[1]
df     = pd.read_csv(root/'data/cleaned_diabetic_data.csv', nrows=3)
resp   = requests.post("http://127.0.0.1:8000/predict",
                       json={"patients": df.to_dict("records")})
print(resp.json())