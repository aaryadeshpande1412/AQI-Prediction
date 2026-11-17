# Trained model files

This folder is intentionally waiting for the trained model artifacts.

After running the model-training notebook, place the fitted scikit-learn pipeline here as:

- `aqi_model.pkl`

If generated, also place the model comparison metrics as:

- `model_evaluation_results.csv`

The dashboard's analytics pages can run without these files, but the AQI Prediction page needs `aqi_model.pkl`.
Do not upload a model unless it was produced by your own training run and matches the 14 input features documented in `dashboard/README.txt`.
