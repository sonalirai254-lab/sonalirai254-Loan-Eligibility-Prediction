# Loan Eligibility Prediction

A clean, beginner-friendly machine learning project that predicts whether a
loan application will be **Approved** or **Rejected** based on applicant
details. It ships with a training script and a minimal Streamlit web app.

## Features

- Loads applicant data from a CSV dataset
- Preprocessing: missing-value imputation, categorical encoding, train/test split
- Trains two models: **Logistic Regression** and **Random Forest**
- Evaluates with accuracy, confusion matrix, and classification report
- Compares both models and saves the best one as `model.pkl`
- Streamlit web app to predict eligibility from a form

## Technologies

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib.

## Project structure

```
loan_eligibility_prediction/
├── loan_data.csv          # dataset
├── loan_prediction.py     # preprocessing + training + evaluation + saves model.pkl
├── app.py                 # Streamlit UI
├── model.pkl              # saved best model (created by loan_prediction.py)
├── README.md
└── requirements.txt
```

## Setup

```bash
cd loan_eligibility_prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```bash
python loan_prediction.py
```

This prints evaluation metrics for both models and writes `model.pkl`
(the better of the two).

## Run the web app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), fill out the
form, and click **Predict** to see **Loan Approved** or **Loan Rejected**.

## Input fields

Gender, Married, Dependents, Education, Self Employed, Applicant Income,
Coapplicant Income, Loan Amount, Loan Amount Term, Credit History,
Property Area.
