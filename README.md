# EMIPredict AI: EMI Eligibility and Maximum Safe EMI Prediction

**Live app:** https://y5rp4cam4l67kt8257hwqm.streamlit.app/

EMIPredict AI is a machine learning project that predicts two things for a loan applicant:

1. **EMI eligibility** (classification): Eligible, High_Risk, or Not_Eligible
2. **Maximum safe monthly EMI** (regression): the amount in INR the applicant can safely repay each month

It includes data cleaning, feature engineering, model training and comparison, experiment tracking with MLflow, and a Streamlit web app.

## Dataset

- File: `emi_prediction_dataset.csv` (about 400,000 records)
- 22 input features covering personal details, income, expenses, credit history, and loan request
- Not included in this repository because of its size. Place it in the project folder to retrain.

## Models

| Task | Models compared | Final model |
|---|---|---|
| Classification | Logistic Regression, Random Forest, XGBoost | XGBoost |
| Regression | Linear Regression, Random Forest, XGBoost | XGBoost |



## Known limitations

- The High_Risk class is harder to predict than the other two classes.
- Interest is not modeled. Monthly EMI is calculated as principal divided by tenure.
- Some salaries in the dataset are above the 200,000 INR input limit, so those applicants cannot be entered in the app.
- Gender is used as a feature, which raises fairness concerns. This is a learning project, not a lending system.

## Project structure
app/app.py Streamlit application
models/ Saved models and feature list (.joblib)
EMIPredict AI - ....ipynb Training notebook: cleaning, EDA, training, MLflow
requirements.txt Python libraries

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Tools used

Python, pandas, scikit-learn, XGBoost, MLflow, Streamlit