from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "svc_trained_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoders.pkl"

FEATURE_ORDER = ["PClass", "Gender", "Sibling", "Embarked"]


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="TS",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    missing = [
        str(path.relative_to(BASE_DIR))
        for path in [MODEL_PATH, ENCODER_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required artifact(s): " + ", ".join(missing)
        )

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(ENCODER_PATH, "rb") as encoder_file:
        encoders = pickle.load(encoder_file)

    return model, encoders


st.title("Titanic Passenger Survival Prediction")
st.write(
    "Enter passenger details using the same simplified categories used in the "
    "training notebook. The app loads the saved SVC model and preprocessing "
    "artifacts instead of retraining."
)

try:
    model, encoders = load_artifacts()
except Exception as exc:
    st.error("The app could not load the trained model files.")
    st.exception(exc)
    st.stop()


col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger class", ["First", "Second", "Third"], index=2)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    sibling = st.selectbox("Siblings/spouses aboard", ["Zero", "One", "Two", "Three"])
    embarked = st.selectbox(
        "Port of embarkation",
        ["Southampton", "Cherbourg", "Queenstown"],
    )


if st.button("Predict survival", type="primary", use_container_width=True):
    user_input = pd.DataFrame(
        {
            "PClass": [pclass],
            "Gender": [gender],
            "Sibling": [sibling],
            "Embarked": [embarked],
        }
    )

    try:
        encoded_input = user_input.copy()
        for column in FEATURE_ORDER:
            encoded_input[column] = encoders[column].transform(user_input[column])

        encoded_input = encoded_input[FEATURE_ORDER]
        prediction_code = int(model.predict(encoded_input)[0])
        prediction_label = encoders["Survived"].inverse_transform([prediction_code])[0]

        if prediction_label == "Yes":
            st.success("Prediction: SURVIVED")
        else:
            st.error("Prediction: NOT SURVIVED")

        with st.expander("Encoded feature vector used by the model"):
            st.dataframe(encoded_input, hide_index=True)

    except Exception as exc:
        st.error("Prediction failed. Please check that the model and encoders match.")
        st.exception(exc)


st.caption(
    "Teaching project for AIC354. This model is trained on a simplified "
    "100-row Titanic sample dataset, so predictions are for learning purposes."
)
