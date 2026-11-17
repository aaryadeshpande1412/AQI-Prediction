# AQI Prediction and Air Quality Analytics Dashboard for Indian Cities

A machine-learning project for analysing historical Air Quality Index (AQI) records across Indian cities and estimating AQI using historical readings.

## Project structure

```text
AQI_Prediction_Analytics_GitHub/
├── data/
│   ├── raw/
│   │   └── AQI_2020_to_2025_October.csv
│   └── processed/
│       ├── cleaned_aqi.csv
│       └── feature_engineered_aqi.csv
├── notebook/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Training.ipynb
│   └── Model_evalution.ipynb
├── models/
│   └── README.md
├── dashboard/
│   ├── app.py
│   ├── run_dashboard.bat
│   └── README.txt
├── requirements.txt
└── .gitignore
```

## Project workflow

1. Understand the raw AQI dataset.
2. Clean and prepare the data.
3. Explore AQI trends and pollutant patterns through EDA.
4. Create date-based and historical AQI features.
5. Train and compare regression models.
6. Run the Streamlit dashboard to explore AQI analytics and predictions.

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Plotly and Streamlit
- Scikit-learn
- Joblib
- Jupyter Notebook

## Setup

Install Python 3.10 or newer, then open a terminal in the repository root and run:

```bash
python -m pip install -r requirements.txt
```

## Run the dashboard

From the repository root:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard uses a light, clean interface with top navigation tabs for Overview, City insights, Pollutants, AQI prediction and Model performance.

Or open `dashboard` and double-click `run_dashboard.bat` on Windows.

The dashboard reads `data/processed/cleaned_aqi.csv`. Its analytics pages can run without a saved model; the AQI Prediction page requires `models/aqi_model.pkl`.

## Model files still to add

The uploaded project files did not include the trained `aqi_model.pkl` or `model_evaluation_results.csv`. After running your model-training/evaluation notebook, copy the saved model to `models/aqi_model.pkl` and the evaluation table to `models/model_evaluation_results.csv` (if generated). If an existing pickle fails with a scikit-learn internal-class error such as `_RemainderColsList`, retrain and save it in the same environment used to run the dashboard. The updated dashboard catches model-load errors so the analytics sections remain usable. See `models/README.md`.

## Dataset note

The CSV files are included because they were supplied for this project. Before making the repository public, check that you are permitted to redistribute the original dataset and that it contains no private or restricted information.

## Important notebook note

Some notebook cells use machine-specific Windows file paths or expect the working directory to be a particular folder. If a notebook reports `FileNotFoundError` after cloning the repository, update that cell's input/output path to the corresponding file under `data/raw` or `data/processed`, or start Jupyter with the expected working directory.
