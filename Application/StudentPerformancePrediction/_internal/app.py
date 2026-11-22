import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction System")
st.caption("Machine Learning project • Explore student data, understand performance patterns, and estimate final scores.")

DATA_PATH = Path(__file__).parent / "data" / "student_performance.csv"
FEATURES = [
    "study_hours_per_day", "attendance_percentage", "previous_score",
    "assignments_completed", "sleep_hours", "internet_access",
    "extracurricular_activities", "parental_support"
]
TARGET = "final_score"
NUMERIC = ["study_hours_per_day", "attendance_percentage", "previous_score",
           "assignments_completed", "sleep_hours"]
CATEGORICAL = ["internet_access", "extracurricular_activities", "parental_support"]

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    return data[required].dropna(subset=[TARGET])

@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL)
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R²": r2_score(y_test, predictions)
    }
    return model, metrics, (y_test, predictions)

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload your own CSV (optional)", type=["csv"])
    st.info("Your CSV must contain the 9 columns shown in the README. The included dataset works immediately.")

try:
    data = load_data(uploaded)
    model, metrics, test_results = train_model(data)
except Exception as exc:
    st.error(f"Could not load/train the project: {exc}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Predict a score", "📁 Dataset"])
with tab1:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(data):,}")
    c2.metric("Average final score", f"{data[TARGET].mean():.1f}/100")
    c3.metric("Average attendance", f"{data['attendance_percentage'].mean():.1f}%")
    c4.metric("Average study time", f"{data['study_hours_per_day'].mean():.1f} hrs/day")
    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("**Distribution of final scores**")
        fig, ax = plt.subplots()
        ax.hist(data[TARGET], bins=18, edgecolor="black")
        ax.set_xlabel("Final score")
        ax.set_ylabel("Number of students")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Study time vs final score**")
        fig, ax = plt.subplots()
        ax.scatter(data["study_hours_per_day"], data[TARGET], alpha=0.45)
        ax.set_xlabel("Study hours per day")
        ax.set_ylabel("Final score")
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("**Model evaluation on a held-out test set (20%)**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
    m2.metric("Root Mean Squared Error", f"{metrics['RMSE']:.2f}")
    m3.metric("R² score", f"{metrics['R²']:.3f}")
    st.caption("MAE/RMSE are in score points. R² measures how much score variation the model explains.")

with tab2:
    st.subheader("Enter student information")
    with st.form("prediction_form"):
        a, b = st.columns(2)
        with a:
            study = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
            attendance = st.slider("Attendance (%)", 0, 100, 80)
            previous = st.slider("Previous score", 0, 100, 70)
            assignments = st.slider("Assignments completed (out of 10)", 0, 10, 7)
        with b:
            sleep = st.slider("Sleep hours per day", 0.0, 12.0, 7.0, 0.5)
            internet = st.selectbox("Internet access", ["Yes", "No"])
            extra = st.selectbox("Extracurricular activities", ["Yes", "No"])
            support = st.selectbox("Parental support", ["Low", "Medium", "High"], index=1)
        submitted = st.form_submit_button("Predict final score", type="primary")
    if submitted:
        sample = pd.DataFrame([{
            "study_hours_per_day": study,
            "attendance_percentage": attendance,
            "previous_score": previous,
            "assignments_completed": assignments,
            "sleep_hours": sleep,
            "internet_access": internet,
            "extracurricular_activities": extra,
            "parental_support": support
        }])
        prediction = float(np.clip(model.predict(sample)[0], 0, 100))
        st.success(f"Estimated final score: {prediction:.1f} / 100")
        if prediction >= 75:
            st.write("Indicative band: High")
        elif prediction >= 50:
            st.write("Indicative band: Moderate")
        else:
            st.write("Indicative band: Needs support")
        st.caption("This is an estimate from patterns in the training data, not a guaranteed result or a judgment of a student.")

with tab3:
    st.subheader("Student dataset")
    st.dataframe(data, use_container_width=True, height=420)
    st.download_button("Download dataset as CSV", data.to_csv(index=False).encode("utf-8"),
                       file_name="student_performance.csv", mime="text/csv")
