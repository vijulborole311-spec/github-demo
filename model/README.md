# Cervical Cancer Risk Factor Analysis

This project explores cervical cancer risk-factor data and includes a Jupyter
notebook and a Streamlit demonstration app using logistic regression.

## Run the Streamlit app

1. Install dependencies with `pip install -r model/requirements.txt`.
2. Place the dataset at `model/data/cervical-cancer_csv (1).csv`.
3. From the repository root, run `streamlit run model/app.py`.

The dataset is not included in this public repository. The app expects the CSV
at the path above. To run the analysis notebook, open
`model/cervical-cancer.ipynb` in Jupyter with the dataset available locally.

The app excludes prior diagnosis and screening-result fields from prediction
inputs to reduce target leakage. Those fields remain visible in the dataset
explorer when present.

This is an educational project, not a medical device, screening tool, diagnosis,
or clinical risk assessment.