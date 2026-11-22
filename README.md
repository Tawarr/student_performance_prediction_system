# Student Performance Prediction System

A practical **Machine Learning** project that analyzes student information and estimates a student's final score (0–100). It includes an interactive Streamlit dashboard, exploratory charts, model evaluation, a prediction form, and CSV upload support.

## 1. Technology used
- Python
- Pandas and NumPy for data handling
- Matplotlib for charts
- Scikit-learn: preprocessing pipeline and Random Forest Regression
- Streamlit for the web interface

## 2. Project structure
```text
student_performance_prediction_system/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
└── data/
    └── student_performance.csv
```

## 3. Run on Windows
Install Python 3.10 or newer. Open Command Prompt in this project folder and run:

```bat
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The app should open in your browser at `http://localhost:8501`. If it does not, copy that address into your browser.

## 4. Input columns
Your uploaded CSV must contain these exact columns:

| Column | Meaning |
|---|---|
| `study_hours_per_day` | Average daily study hours |
| `attendance_percentage` | Attendance from 0 to 100 |
| `previous_score` | Previous exam score from 0 to 100 |
| `assignments_completed` | Completed assignments, 0 to 10 |
| `sleep_hours` | Average daily sleep hours |
| `internet_access` | Yes or No |
| `extracurricular_activities` | Yes or No |
| `parental_support` | Low, Medium, or High |
| `final_score` | Actual final score used as the prediction target |

The included CSV is a **synthetic demonstration dataset** generated for this project. It is suitable for demonstrating the full pipeline, but it is not real student-record evidence. For a serious academic evaluation, replace it with a properly sourced, documented, permission-cleared real dataset and retrain the model.

## 5. How it works
1. Load the CSV dataset.
2. Separate input features from `final_score`.
3. Split data into training (80%) and testing (20%).
4. Impute missing numeric/categorical values and one-hot encode categorical features.
5. Train a Random Forest Regressor.
6. Evaluate using MAE, RMSE, and R².
7. Use the trained pipeline to estimate scores for new inputs.

## 6. Notes
- This is a regression task because the model predicts a numeric score.
- The dashboard metrics depend on the dataset supplied.
- Predictions are estimates, not guarantees. Do not use them as the sole basis for high-stakes academic decisions.
