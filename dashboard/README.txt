AQI DASHBOARD - SETUP INSTRUCTIONS
==================================

FOLDER LAYOUT
-------------
This dashboard is part of the repository. It reads:
  ../data/processed/cleaned_aqi.csv
  ../models/aqi_model.pkl
  ../models/model_evaluation_results.csv (optional)

INSTALL
-------
From the repository root:
  python -m pip install -r requirements.txt

RUN
---
From the repository root:
  python -m streamlit run dashboard/app.py

Or open this folder and double-click run_dashboard.bat.

NAVIGATION
----------
The dashboard uses a bright, light theme and five visible top tabs:
Overview, City insights, Pollutants, AQI prediction, Model performance.

MODEL COMPATIBILITY
-------------------
The AQI prediction page requires models/aqi_model.pkl, saved as a fitted
scikit-learn Pipeline with these 14 input features:

City, Year, Month, Day, DayOfYear, DayOfWeek, Quarter,
Previous_AQI, Previous_2_AQI, Previous_3_AQI, Previous_7_AQI,
Average_3, Average_7, Recent_Change

If the model shows an error such as "_RemainderColsList", it was likely
serialized with a different scikit-learn version. The app now catches this
error and keeps all analytics pages working. Retrain the model in the same
environment used to run the dashboard, then replace models/aqi_model.pkl.
A pickle file cannot safely be made compatible merely by editing app.py.

The dashboard is based on historical data and does not show live AQI.
