# DeadlineSense - Streamlit Prototype

MSc Advanced Computer Science Masters Project (7COM1039)
Student: Farhan Ali (23104223)

DeadlineSense is a free, effort-aware deadline reminder for a single software
task. It predicts the total effort of a task from its title and description
(model trained on 20,862 real open-source tasks from the JOSSE dataset),
combines the prediction with the user's progress and daily working hours,
and shows an explainable ON TRACK / AT RISK warning.

## Files

- `deadlinesense_app.py` - the Streamlit web application
- `test_deadlinesense.py` - unit tests for the core logic (proposal Section 6.8)
- `requirements.txt` - Python dependencies
- `deadlinesense_vectorizer.joblib` - trained TF-IDF vectorizer (from the project notebook)
- `deadlinesense_model.joblib` - trained Ridge regression model (from the project notebook)

## Setup (one time)

1. Install Python 3.10+ from https://www.python.org/downloads/
   (on Windows, tick "Add Python to PATH" during installation)
2. Open a terminal / command prompt in this folder
3. Install dependencies:

   pip install -r requirements.txt

## Run the app

   streamlit run deadlinesense_app.py

A browser tab opens at http://localhost:8501 with the app.
Everything runs locally - no data leaves your computer.

## Run the unit tests

   python -m pytest test_deadlinesense.py -v

All 15 tests should pass. Each test uses a worked example whose correct
output was calculated by hand in advance, as specified in the proposal.

## Model summary

- Data: 20,862 cleaned tasks from 363 open-source projects (JOSSE dataset)
- Pipeline: TF-IDF (10,000 features, 1-2 grams) + Ridge regression (alpha=10)
  on log-transformed effort hours
- Validation: project-grouped 5-fold cross-validation
- Accuracy: MAE 2.67h; RMSE 5.56h (6.3% better than a median baseline,
  statistically significant, p < 0.001)