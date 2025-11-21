import requests
import pandas as pd

API_Url = "http://localhost:8000"

def predict_churn(payload: dict):
    url = f"{API_Url}/churn/predict"
    try:
        r = requests.post(url, json={"customer_data": payload})
        print("\n---- DEBUG API RESPONSE ----")
        print("STATUS:", r.status_code)
        print("TEXT:", repr(r.text))
        print("----------------------------\n")
        
        if r.status_code == 200:
            return r.json()
        else:
            try:
                error_detail = r.json().get("detail", r.text)
            except:
                error_detail = r.text
            return {"error": f"API returned {r.status_code}", "detail": error_detail}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API", "detail": "Make sure FastAPI server is running on http://localhost:8000"}
    except Exception as e:
        return {"error": "Request failed", "detail": str(e)}

def explain_churn(payload: dict):
    url = f"{API_Url}/churn/explain"
    try:
        r = requests.post(url, json={"customer_data": payload})
        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"API returned {r.status_code}", "detail": r.text}
    except Exception as e:
        return {"error": "Request failed", "detail": str(e)}

def batch_churn(df: pd.DataFrame):
    url = f"{API_Url}/churn/batch"
    data = {"records": df.to_dict(orient="records")}
    try:
        r = requests.post(url, json=data)
        if r.status_code == 200:
            return r.json()
        else:
            try:
                error_detail = r.json().get("detail", r.text)
            except:
                error_detail = r.text
            return {"error": f"API returned {r.status_code}", "detail": error_detail}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API", "detail": "Make sure FastAPI server is running on http://localhost:8000"}
    except Exception as e:
        return {"error": "Request failed", "detail": str(e)}

def predict_clv(payload: dict):
    url = f"{API_Url}/clv/predict"
    try:
        r = requests.post(url, json={"customer_data": payload})
        if r.status_code == 200:
            return r.json()
        else:
            try:
                error_detail = r.json().get("detail", r.text)
            except:
                error_detail = r.text
            return {"error": f"API returned {r.status_code}", "detail": error_detail}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API", "detail": "Make sure FastAPI server is running on http://localhost:8000"}
    except Exception as e:
        return {"error": "Request failed", "detail": str(e)}

def get_segment(payload: dict):
    url = f"{API_Url}/segments/predict"
    try:
        r = requests.post(url, json={"customer_data": payload})
        if r.status_code == 200:
            return r.json()
        else:
            try:
                error_detail = r.json().get("detail", r.text)
            except:
                error_detail = r.text
            return {"error": f"API returned {r.status_code}", "detail": error_detail}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API", "detail": "Make sure FastAPI server is running on http://localhost:8000"}
    except Exception as e:
        return {"error": "Request failed", "detail": str(e)}