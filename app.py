## Streamlit app for predicting if a person will be on leave or not

# Importing libraries

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

MODEL_COLUMNS = ["degrees_from_mean", "longitude", "latitude", "max_temp", "min_temp"]
TYPE_CODE_MAPPINGS = {3: "Weak Hot", 2: "Weak Cold", 1: "Strong Cold", 4: "Strong Hot"}

st.set_page_config(layout="wide")

# App title
st.title("Weather Anomalies Detection")

# Add an upload button for data upload
st.subheader("1. Train and Evaluate the Model")
uploaded_file = st.file_uploader("Upload data in CSV to train the model", type=["csv"])

# Create a dataframe from the uploaded file
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Get input from user on the number of rows to display
    num_rows = st.number_input(
        "Enter the number of rows to display", min_value=0, max_value=30, value=5
    )
    # Show the top 5 row of the dataframe
    st.header("Data Sample")
    st.dataframe(data.head(num_rows))


# create a function to plot categorical variables
def plot_num(data, num_var):
    st.header("Plots of " + num_var)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1.hist(data[num_var].dropna(), bins=30, color="blue", edgecolor="black")
    ax1.set_title(f"Histogram of {num_var}")
    ax1.set_xlabel(num_var)
    ax1.set_ylabel("Frequency")

    # Box Plot
    sns.boxplot(data=data, x=num_var, ax=ax2, color="blue")
    ax2.set_title(f"Box Plot of {num_var}")
    ax2.set_xlabel(num_var)

    st.pyplot(fig)


def clean_data(data):
    return data


if uploaded_file:
    # Get a list of all the columns in the dataframe
    columns = data.select_dtypes("number").columns

    # Create a dropdown where user can select the column to plot
    num_var = st.selectbox("Select a column to plot", columns)

    # Plot the selected column
    plot_num(data, num_var)

    # Drop irrelevant features for the model development
    df_model = data.drop(["id", "station_name", "serialid", "date_str"], axis=1)

    # Encoding of the output variable
    df_model["type"] = df_model.type.map(
        {"Weak Hot": 3, "Weak Cold": 2, "Strong Cold": 1, "Strong Hot": 4}
    )

    # show the top 3 our updated dataframe
    st.header("Clean and encode output variable")
    st.dataframe(df_model.head(3))

    # Create our target and features
    X = df_model[MODEL_COLUMNS]
    y = df_model.type

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Model training
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")

    # Model evaluation
    y_pred = model.predict(X_test)

    # Print the classification report
    st.header("Classification Report")
    class_report = classification_report(y_test, y_pred)
    st.text(class_report)

st.subheader("2. Dataset Based Prediction")
prediction_file = st.file_uploader(
    "Upload a dataset for prediction",
    type=["csv"],
    key="prediction",
)
st.markdown(
    """
    Ensure the dataset has the following columns:
    
    - |degrees_from_mean | longitude | latitude | max_temp | min_temp|
      |------------------|-----------|----------|----------|---------|            
    """
)

if prediction_file:
    model = joblib.load("model.pkl")
    data = pd.read_csv(prediction_file).sample(100)
    df_pred = data[MODEL_COLUMNS]
    prediction = model.predict(df_pred)
    data["predicted_type"] = prediction
    data["predicted_type"] = data.predicted_type.map(TYPE_CODE_MAPPINGS)

    # Get user input on the number of rows to display
    num_rows_pred = st.number_input(
        "Enter the number of rows to display", min_value=0, max_value=50, value=5
    )

    # Show the top 5 rows of the dataframe
    st.subheader("Predictions")
    st.dataframe(data.head(num_rows_pred))

st.subheader("3. Single Record Based Prediction")
mean_deviation = st.number_input("Degrees From Mean", value=20.46)
latitude = st.number_input("Longitude", value=-119.5128)
longitude = st.number_input("Latitude", value=37.0919)
max_temp = st.number_input("Max Temperature", value=25.6)
min_temp = st.number_input("Min Temperature", value=12.8)

if st.button("Predict"):
    df_record = pd.DataFrame(
        [
            {
                "degrees_from_mean": mean_deviation,
                "longitude": longitude,
                "latitude": latitude,
                "max_temp": max_temp,
                "min_temp": min_temp,
            }
        ]
    )
    model = joblib.load("model.pkl")
    record_pred = model.predict(df_record)
    df_record["predicted_type"] = record_pred
    df_record["predicted_type"] = df_record.predicted_type.map(TYPE_CODE_MAPPINGS)
    st.dataframe(df_record)
