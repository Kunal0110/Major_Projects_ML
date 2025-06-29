# smart_health – 30-Day Hospital Readmission Pipeline

```bash
# one-liner setup
pip install -r requirements.txt
python -m src.data.clean           # ➜ data/processed/cleaned_diabetic_data.csv
python -m src.models.search        # ➜ models/best_model.pkl
pytest -q                          # smoke-test
uvicorn src.api.app:app --reload   # REST service
streamlit run app/dashboard.py     # dashboard