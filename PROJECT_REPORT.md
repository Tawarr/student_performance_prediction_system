# PROJECT REPORT
## Student Performance Prediction System
**Domain:** Machine Learning (ML)

### 1. Abstract
The Student Performance Prediction System is a machine-learning application that analyzes academic and lifestyle-related student attributes and estimates a student's final examination score. The project uses Python, Pandas, Scikit-learn, and Streamlit. It provides a dashboard for exploring the dataset, reports model evaluation metrics, and allows users to enter student details to obtain an estimated score.

### 2. Problem Statement
Educational performance is influenced by several factors, including study time, attendance, previous academic performance, assignment completion, and learning support. Reviewing these factors manually can be time-consuming. This project demonstrates how a supervised machine-learning regression model can learn patterns from historical records and estimate a numeric final score.

### 3. Objectives
- Explore student performance data using summary statistics and charts.
- Build a supervised regression model to estimate final scores.
- Evaluate the model on data not used during training.
- Provide an easy-to-use interface for predictions.
- Allow a compatible CSV dataset to be uploaded.

### 4. Dataset Description
The included demonstration dataset contains 1,200 synthetic student records. Features include study hours, attendance percentage, previous score, assignments completed, sleep hours, internet access, extracurricular participation, and parental support. The target variable is `final_score`, measured from 0 to 100.

**Dataset limitation:** The included data is synthetic and was created to make the project runnable immediately. It does not represent actual students and cannot establish real-world predictive accuracy. Replace it with a documented real dataset before making claims about real student outcomes.

### 5. Methodology
1. **Data loading:** Read the CSV file using Pandas.
2. **Preprocessing:** Fill missing numeric values with the median and categorical values with the most frequent category.
3. **Encoding:** Convert categorical values into numeric indicator columns using One-Hot Encoding.
4. **Splitting:** Divide the dataset into 80% training and 20% testing records.
5. **Model training:** Train a Random Forest Regressor with 250 trees.
6. **Evaluation:** Calculate Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².
7. **Deployment:** Present the dashboard and prediction form through Streamlit.

### 6. Algorithms and Evaluation
Random Forest Regression combines predictions from multiple decision trees. It can model non-linear relationships and interactions between input features.

- **MAE:** Average absolute difference between actual and predicted scores. Lower is better.
- **RMSE:** Square root of average squared prediction error. Lower is better and larger errors receive more penalty.
- **R²:** Measures the proportion of target variation explained by the model. A value closer to 1 is generally better; it can be negative on poor predictions.

The application calculates these metrics dynamically from the selected dataset; do not copy a metric into the report without running the application and recording the displayed result.

### 7. System Modules
- Dataset loading and validation
- Data overview and visualization
- Data preprocessing
- Model training and testing
- Performance evaluation
- Student score prediction
- Dataset display and download

### 8. Expected Output
The application displays the number of records, average score, average attendance, and average study time. It visualizes score distribution and the relationship between study time and final score. The prediction page accepts student details and returns an estimated final score and an indicative performance band.

### 9. Conclusion
The project demonstrates a complete supervised machine-learning workflow, from data loading and preprocessing to model evaluation and a simple interactive deployment. It can be extended with real educational datasets, classification of risk categories, explainability tools, and comparisons between multiple regression algorithms.

### 10. Future Scope
- Use a larger, real, ethically sourced dataset.
- Compare Random Forest with Linear Regression, Decision Tree, and Gradient Boosting.
- Add feature-importance and explainability views.
- Add student-wise progress tracking over multiple terms.
- Evaluate fairness and error rates across relevant groups before any real deployment.

### 11. References
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/
- Pandas documentation: https://pandas.pydata.org/docs/
