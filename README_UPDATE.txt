STUDENT PERFORMANCE PREDICTION SYSTEM - UPDATE

WHAT THIS UPDATE ADDS
- Keeps support for the original student_performance.csv dataset.
- Adds support for StudentPerformanceFactors.csv.
- Builds the prediction form dynamically from the selected dataset's input columns.
- Uses the selected dataset's actual target column (final_score or Exam_Score).
- Supports additional CSV files, with manual target and input-column selection for unknown formats.
- Imputes missing numeric and categorical predictor values during model training.

INSTALL THE UPDATE
1. Close the running Streamlit application.
2. Make a backup copy of your current app.py.
3. Extract this ZIP into your existing project folder (the folder containing app.py).
4. Allow the new app.py to replace the old app.py.
5. The ZIP adds data/StudentPerformanceFactors.csv. Keep your existing
   data/student_performance.csv file; this update does not replace it.
6. Start the app from PowerShell in the project folder:

   .venv\Scripts\python.exe -m streamlit run app.py

7. In the sidebar, choose either:
   - Original project dataset
   - Student Performance Factors
   - Upload another CSV

IMPORTANT NOTES
- In StudentPerformanceFactors.csv, Hours_Studied means hours studied per week.
- Parental_Involvement is not identical to Parental_Support; the Factors model
  treats it as its own separate feature.
- StudentPerformanceFactors.csv has no assignments-completed column. The model
  for that dataset therefore does not use assignments_completed.
- The data is used as supplied. This CSV includes an Exam_Score value of 101;
  the app does not silently alter that value.
- Predictions from different datasets are not automatically directly comparable
  unless their target scales, populations, and evaluation methods are aligned.
