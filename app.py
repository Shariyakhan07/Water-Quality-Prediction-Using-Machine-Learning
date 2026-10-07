import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide"
)


@st.cache_resource
def train_model():

    df = pd.read_csv("water_quality_dataset.csv")

    df = df.drop_duplicates()

    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        if column != "Potability":
            df[column] = df[column].fillna(df[column].median())

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_probability)
    }

    return model, metrics


model, metrics = train_model()


st.title("💧 Water Quality Prediction")
st.write(
    "Predict whether a water sample is potable based on its quality parameters."
)

st.divider()

st.subheader("Enter Water Quality Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0
    )

    hardness = st.number_input(
        "Hardness",
        min_value=0.0,
        value=190.0
    )

    solids = st.number_input(
        "Solids",
        min_value=0.0,
        value=22000.0
    )

with col2:
    chloramines = st.number_input(
        "Chloramines",
        min_value=0.0,
        value=7.0
    )

    sulfate = st.number_input(
        "Sulfate",
        min_value=0.0,
        value=330.0
    )

    conductivity = st.number_input(
        "Conductivity",
        min_value=0.0,
        value=420.0
    )

with col3:
    organic_carbon = st.number_input(
        "Organic Carbon",
        min_value=0.0,
        value=14.0
    )

    trihalomethanes = st.number_input(
        "Trihalomethanes",
        min_value=0.0,
        value=65.0
    )

    turbidity = st.number_input(
        "Turbidity",
        min_value=0.0,
        value=3.8
    )


st.divider()

if st.button("🔍 Predict Water Quality", use_container_width=True):

    sample = np.array([[
        ph,
        hardness,
        solids,
        chloramines,
        sulfate,
        conductivity,
        organic_carbon,
        trihalomethanes,
        turbidity
    ]])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("💧 POTABLE — Water is predicted to be suitable for drinking.")
    else:
        st.error("⚠️ NOT POTABLE — Water is predicted to be unsuitable for drinking.")

    st.metric(
        "Potability Probability",
        f"{probability * 100:.2f}%"
    )


st.divider()

st.subheader("Model Performance")

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")

with metric_col2:
    st.metric("Precision", f"{metrics['Precision']:.2%}")

with metric_col3:
    st.metric("Recall", f"{metrics['Recall']:.2%}")

with metric_col4:
    st.metric("F1 Score", f"{metrics['F1 Score']:.2%}")

with metric_col5:
    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2%}")


st.divider()

st.subheader("About the Project")

st.write(
    "This machine learning project uses water quality parameters "
    "to predict water potability. The Random Forest Classifier "
    "is trained using cleaned water quality data."
)

st.write(
    "Parameters used: pH, Hardness, Solids, Chloramines, Sulfate, "
    "Conductivity, Organic Carbon, Trihalomethanes and Turbidity."
)

st.warning(
    "This application provides a machine-learning prediction and "
    "should not replace certified laboratory  water testing."
)