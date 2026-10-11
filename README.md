# Titanic Passenger Survival Prediction

This project is an AIC354 machine learning assignment based on the instructor's
Titanic passenger survival notebook. It trains a Support Vector Classifier (SVC)
on a simplified Titanic dataset and deploys the saved model with Streamlit.

## Project Files

- `titanic_survival_prediction.ipynb` - completed and executable notebook.
- `data/sample-data.csv` - required simplified Titanic dataset.
- `models/svc_trained_model.pkl` - trained SVC model.
- `models/label_encoders.pkl` - preprocessing encoders used by the app.
- `models/training_metadata.json` - feature order, categories, and metrics.
- `app.py` - Streamlit web app.
- `requirements.txt` - Python dependencies for notebook execution and deployment.

The original instructor notebook is preserved as
`Titanic_Passenger_Survival_Recommended.ipynb`.

## Model Summary

The notebook follows the instructor's workflow:

1. Load the simplified Titanic dataset.
2. Inspect columns, distributions, and missing values.
3. Encode categorical features with label encoders.
4. Split the dataset into training and testing data.
5. Train an `sklearn.svm.SVC(gamma="auto", random_state=0)` classifier.
6. Evaluate the classifier on the testing data.
7. Save the trained model and preprocessing artifacts.

Held-out test accuracy from the executed notebook is 0.80.

## Run Locally

Create or activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

The app loads files from `models/` and does not retrain the model at startup.

## Deployment

Deploy on Streamlit Community Cloud using:

- Repository: `https://github.com/Ab-Wb-Sid/Titanic-Survival-Prediction`
- Branch: `main`
- App file: `app.py`

After deployment, add the live Streamlit URL to the first Markdown cell of the
notebook before final classroom submission.
