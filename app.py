import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# App setup and dataset locations
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ORIGINAL_DATA_PATH = DATA_DIR / "student_performance.csv"
FACTORS_DATA_PATH = DATA_DIR / "StudentPerformanceFactors.csv"

st.title("🎓 Student Performance Prediction System")
st.caption(
    "Machine Learning project • Explore student data, understand performance "
    "patterns, compare datasets, and estimate student scores."
)


# -----------------------------------------------------------------------------
# Column labels and automatic target detection
# -----------------------------------------------------------------------------
COLUMN_LABELS = {
    "study_hours_per_day": "Study hours per day",
    "hours_studied": "Hours studied per week",
    "attendance_percentage": "Attendance (%)",
    "attendance": "Attendance (%)",
    "previous_score": "Previous score",
    "previous_scores": "Previous scores",
    "assignments_completed": "Assignments completed",
    "sleep_hours": "Sleep hours per day",
    "internet_access": "Internet access",
    "extracurricular_activities": "Extracurricular activities",
    "parental_support": "Parental support",
    "parental_involvement": "Parental involvement",
    "access_to_resources": "Access to resources",
    "motivation_level": "Motivation level",
    "tutoring_sessions": "Tutoring sessions",
    "family_income": "Family income",
    "teacher_quality": "Teacher quality",
    "school_type": "School type",
    "peer_influence": "Peer influence",
    "physical_activity": "Physical activity",
    "learning_disabilities": "Learning disabilities",
    "parental_education_level": "Parental education level",
    "distance_from_home": "Distance from home",
    "gender": "Gender",
    "final_score": "Final score",
    "exam_score": "Exam score",
}

TARGET_ALIASES = {
    "finalscore",
    "examscore",
    "finalgrade",
    "finalmark",
    "finalmarks",
    "targetscore",
    "target",
    "score",
}


def normalized_name(name):
    """Return a column name with punctuation, spaces and case removed."""
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def display_name(column):
    """Human-readable label for a column."""
    key = str(column).strip().lower()
    return COLUMN_LABELS.get(key, str(column).replace("_", " ").replace(".", " ").title())


def detect_target_column(columns):
    """Find a likely score/grade column; return None if no safe match exists."""
    normalized = {normalized_name(col): col for col in columns}
    for alias in TARGET_ALIASES:
        if alias in normalized:
            return normalized[alias]
    return None


def prepare_dataset(raw_data):
    """Clean basic CSV issues and identify a known or generic dataset profile."""
    data = raw_data.copy()
    data.columns = [str(col).strip() for col in data.columns]

    # Remove completely blank columns and common exported row-index columns.
    data = data.dropna(axis=1, how="all")
    index_like = [
        col for col in data.columns
        if normalized_name(col) in {"unnamed0", "index", "rowindex"}
    ]
    if index_like:
        data = data.drop(columns=index_like)

    # Detect the supplied original project dataset and standardize its names.
    legacy_aliases = {
        "study_hours_per_day": ["study_hours_per_day", "study_hours"],
        "attendance_percentage": ["attendance_percentage", "attendance_pct"],
        "previous_score": ["previous_score", "previous_grade"],
        "assignments_completed": ["assignments_completed"],
        "sleep_hours": ["sleep_hours", "sleep_hours_per_day"],
        "internet_access": ["internet_access"],
        "extracurricular_activities": ["extracurricular_activities"],
        "parental_support": ["parental_support"],
        "final_score": ["final_score", "final_grade", "final_marks"],
    }
    norm_to_original = {normalized_name(col): col for col in data.columns}
    rename_map = {}
    legacy_complete = True
    for standard_name, aliases in legacy_aliases.items():
        found = None
        for alias in aliases:
            key = normalized_name(alias)
            if key in norm_to_original:
                found = norm_to_original[key]
                break
        if found is None:
            legacy_complete = False
            break
        if found != standard_name:
            rename_map[found] = standard_name

    if legacy_complete:
        data = data.rename(columns=rename_map)
        target = "final_score"
        features = [
            "study_hours_per_day",
            "attendance_percentage",
            "previous_score",
            "assignments_completed",
            "sleep_hours",
            "internet_access",
            "extracurricular_activities",
            "parental_support",
        ]
        profile = "Original Student Performance Dataset"
        return data, features, target, profile

    # Recognize the supplied StudentPerformanceFactors.csv.
    norm_to_original = {normalized_name(col): col for col in data.columns}
    if "examscore" in norm_to_original and "hoursstudied" in norm_to_original:
        target = norm_to_original["examscore"]
        features = [col for col in data.columns if col != target]
        profile = "Student Performance Factors"
        return data, features, target, profile

    # For other CSVs, use a recognizable target if available. The user can
    # choose a different target and predictor columns in the sidebar.
    target = detect_target_column(data.columns)
    if target is None and len(data.columns) >= 2:
        target = data.columns[-1]
    features = [col for col in data.columns if col != target]
    profile = "Custom CSV dataset"
    return data, features, target, profile


def clean_for_training(data, features, target):
    """Make the selected target numeric and remove unusable predictor columns."""
    work = data[features + [target]].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])

    # Empty strings should be treated as missing values.
    for column in features:
        if pd.api.types.is_object_dtype(work[column]) or pd.api.types.is_string_dtype(work[column]):
            work[column] = work[column].replace(r"^\s*$", np.nan, regex=True)

    usable_features = [col for col in features if not work[col].isna().all()]
    dropped_features = [col for col in features if col not in usable_features]
    if not usable_features:
        raise ValueError("No usable input columns remain after removing empty columns.")
    if len(work) < 10:
        raise ValueError(
            f"Only {len(work)} rows have a valid numeric target. At least 10 rows are needed to train and evaluate the model."
        )
    return work[usable_features + [target]], usable_features, dropped_features


@st.cache_resource(show_spinner="Training the machine-learning model...")
def train_model(data, features, target):
    X = data[features].copy()
    y = data[target].copy()

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in features if col not in numeric_features]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R²": r2_score(y_test, predictions) if len(y_test) > 1 else float("nan"),
    }
    return model, metrics, (y_test, predictions), numeric_features, categorical_features


def numeric_default(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0, -1.0, 1.0, 1.0, False
    low = float(values.min())
    high = float(values.max())
    median = float(values.median())
    is_integer = pd.api.types.is_integer_dtype(series.dtype)
    if low == high:
        low -= 1
        high += 1
    step = 1.0 if is_integer else max((high - low) / 100.0, 0.1)
    return median, low, high, step, is_integer


def make_prediction_form(features, data, numeric_features, categorical_features):
    """Build the prediction form from the columns used by the selected model."""
    values = {}
    with st.form("dynamic_prediction_form"):
        left, right = st.columns(2)
        for index, feature in enumerate(features):
            container = left if index % 2 == 0 else right
            label = display_name(feature)
            series = data[feature]
            with container:
                if feature in numeric_features:
                    default, low, high, step, is_integer = numeric_default(series)
                    if is_integer:
                        value = st.number_input(
                            label,
                            min_value=int(math.floor(low)),
                            max_value=int(math.ceil(high)),
                            value=int(round(default)),
                            step=1,
                            key=f"prediction_{index}_{feature}",
                        )
                    else:
                        value = st.number_input(
                            label,
                            min_value=float(low),
                            max_value=float(high),
                            value=float(default),
                            step=float(step),
                            format="%.2f",
                            key=f"prediction_{index}_{feature}",
                        )
                else:
                    choices = series.dropna().astype(str).value_counts().index.tolist()
                    if not choices:
                        choices = ["Unknown"]
                    if len(choices) <= 40:
                        value = st.selectbox(
                            label,
                            choices,
                            key=f"prediction_{index}_{feature}",
                        )
                    else:
                        common = choices[0]
                        value = st.text_input(
                            label,
                            value=common,
                            key=f"prediction_{index}_{feature}",
                        )
                values[feature] = value

        submitted = st.form_submit_button("Predict score", type="primary")
    return submitted, values


# -----------------------------------------------------------------------------
# Select the data source
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data source")
    available_datasets = {}
    if ORIGINAL_DATA_PATH.exists():
        available_datasets["Original project dataset"] = ORIGINAL_DATA_PATH
    if FACTORS_DATA_PATH.exists():
        available_datasets["Student Performance Factors"] = FACTORS_DATA_PATH

    options = list(available_datasets.keys()) + ["Upload another CSV"]
    selected_source = st.selectbox("Choose a dataset", options)
    uploaded_file = None
    if selected_source == "Upload another CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

try:
    if selected_source == "Upload another CSV":
        if uploaded_file is None:
            st.info("Upload a CSV file to continue.")
            st.stop()
        raw_data = pd.read_csv(uploaded_file)
    else:
        raw_data = pd.read_csv(available_datasets[selected_source])

    raw_data, detected_features, detected_target, profile_name = prepare_dataset(raw_data)
    if detected_target is None:
        st.error("I couldn't identify a score/grade column. Please upload a CSV with a numeric target column.")
        st.stop()

    # Known datasets are configured automatically. For an unfamiliar CSV,
    # expose manual target and input-column selection.
    if profile_name == "Custom CSV dataset":
        with st.sidebar:
            st.divider()
            st.subheader("Configure custom dataset")
            target_options = list(raw_data.columns)
            target_index = target_options.index(detected_target) if detected_target in target_options else len(target_options) - 1
            chosen_target = st.selectbox("Column to predict", target_options, index=target_index)
            feature_options = [col for col in raw_data.columns if col != chosen_target]
            chosen_features = st.multiselect(
                "Columns used to make predictions",
                feature_options,
                default=feature_options,
            )
        target_column = chosen_target
        feature_columns = chosen_features
    else:
        target_column = detected_target
        feature_columns = detected_features

    if not feature_columns:
        st.error("Select at least one input feature in the sidebar.")
        st.stop()

    training_data, feature_columns, dropped_features = clean_for_training(
        raw_data, feature_columns, target_column
    )
    if dropped_features:
        st.warning("Ignored completely empty input columns: " + ", ".join(dropped_features))

    model, metrics, test_results, numeric_features, categorical_features = train_model(
        training_data, tuple(feature_columns), target_column
    )

except Exception as exc:
    st.error(f"Could not load/train the project: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Main application tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Predict a score", "📁 Dataset"])

with tab1:
    st.subheader("Dataset overview")
    st.caption(f"Active dataset: **{profile_name}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(training_data):,}")
    c2.metric(f"Average {display_name(target_column)}", f"{training_data[target_column].mean():.1f}")

    normalized_features = {normalized_name(col): col for col in feature_columns}
    attendance_col = normalized_features.get("attendancepercentage") or normalized_features.get("attendance")
    study_col = (
        normalized_features.get("studyhoursperday")
        or normalized_features.get("hoursstudied")
        or normalized_features.get("studyhours")
    )

    if attendance_col:
        c3.metric("Average attendance", f"{training_data[attendance_col].mean():.1f}%")
    else:
        c3.metric("Input features", f"{len(feature_columns)}")

    if study_col:
        study_label = display_name(study_col)
        unit = "hrs/week" if normalized_name(study_col) == "hoursstudied" else "hrs/day"
        c4.metric("Average study time", f"{training_data[study_col].mean():.1f} {unit}")
    else:
        c4.metric("Numeric features", f"{len(numeric_features)}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Distribution of {display_name(target_column).lower()}**")
        fig, ax = plt.subplots()
        ax.hist(training_data[target_column], bins=18, edgecolor="black")
        ax.set_xlabel(display_name(target_column))
        ax.set_ylabel("Number of students")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Input feature vs score**")
        plot_feature = study_col
        if plot_feature is None and numeric_features:
            correlations = training_data[numeric_features + [target_column]].corr(numeric_only=True)[target_column].drop(target_column)
            if not correlations.empty:
                plot_feature = correlations.abs().idxmax()
        if plot_feature is not None and plot_feature in training_data.columns:
            fig, ax = plt.subplots()
            ax.scatter(training_data[plot_feature], training_data[target_column], alpha=0.4)
            ax.set_xlabel(display_name(plot_feature))
            ax.set_ylabel(display_name(target_column))
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No numeric input feature is available for a scatter plot.")

    st.markdown("**Model evaluation on a held-out test set (20%)**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
    m2.metric("Root Mean Squared Error", f"{metrics['RMSE']:.2f}")
    m3.metric("R² score", f"{metrics['R²']:.3f}")
    st.caption(
        "MAE/RMSE are measured in the target column's score units. R² measures "
        "how much target variation the model explains on the test split."
    )

    with st.expander("Dataset quality checks"):
        st.write(f"Rows used for training/evaluation: {len(training_data):,}")
        st.write(f"Input features used: {len(feature_columns)}")
        st.write(f"Duplicate rows: {int(training_data.duplicated().sum()):,}")
        st.write("Missing values in selected columns (imputed during training):")
        missing_counts = training_data[feature_columns].isna().sum()
        st.dataframe(missing_counts[missing_counts > 0].rename("Missing values").to_frame(), use_container_width=True)
        st.write(
            f"Target range: {training_data[target_column].min():.1f} to "
            f"{training_data[target_column].max():.1f}"
        )

with tab2:
    st.subheader("Enter student information")
    st.caption(
        "The fields below are generated from the selected dataset. "
        "The application uses only the input columns selected for that model."
    )
    submitted, entered_values = make_prediction_form(
        feature_columns, training_data, numeric_features, categorical_features
    )
    if submitted:
        sample = pd.DataFrame([entered_values], columns=feature_columns)
        prediction = float(model.predict(sample)[0])
        st.success(f"Estimated {display_name(target_column)}: {prediction:.1f}")
        st.caption(
            "This is a statistical estimate learned from the selected dataset, "
            "not a guaranteed result or a judgment of a student."
        )

with tab3:
    st.subheader("Student dataset")
    st.caption(f"Showing the selected dataset: {profile_name}")
    st.dataframe(raw_data, use_container_width=True, height=420)
    st.download_button(
        "Download active dataset as CSV",
        data=raw_data.to_csv(index=False).encode("utf-8"),
        file_name="active_student_dataset.csv",
        mime="text/csv",
    )
