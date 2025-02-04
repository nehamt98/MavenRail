import os
import pandas as pd
from pathlib import Path
import pickle

from loguru import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_dataset(filename: str) -> pd.DataFrame:
    """
    Loads a dataset from the 'datasets' directory.

    Args:
        filename (str): Name of the CSV file to load.

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """
    base_path = Path().resolve()

    try:
        df = pd.read_csv(os.path.join(base_path, "datasets", filename), delimiter=";")
    except Exception as e:
        logger.error(e)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation and data formatting are carried out for the columns - Railcard, Reason for Delay and Date of Journey

    Args:
        df (pd.DataFrame): The loaded dataset

    Returns:
        pd.DataFrame: The cleaned dataset
    """
    df["Railcard"] = df["Railcard"].fillna("None")
    df["Reason for Delay"] = df["Reason for Delay"].fillna("Not Delayed")
    df["Date of Journey"] = pd.to_datetime(df["Date of Journey"], format="%d/%m/%Y")

    return df


def process_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    A new column 'DelayInMinutes' is calculated based on the Actual ans Scheduled arrival times

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """
    # Subtract Actual Arrival with Scheduled Arrival
    df["Actual Arrival Time"] = pd.to_timedelta(df["Actual Arrival Time"])
    df["Arrival Time"] = pd.to_timedelta(df["Arrival Time"])

    # Calculate the delay, accounting for the potential next day overflow
    df["DelayInMinutes"] = (
        df["Actual Arrival Time"] - df["Arrival Time"]
    ).dt.total_seconds() / 60
    # If delay is negative (next day), add 24 hours (1 day = 1440 minutes)
    df["DelayInMinutes"] = df["DelayInMinutes"].apply(
        lambda x: x + 1440 if x < 0 else x
    )  # When journey status is on time, set it to null. When journey status is cancelled, it is already null.

    df["DelayInMinutes"] = df["DelayInMinutes"].fillna(0)

    return df


def transformation(df: pd.DataFrame):
    """
    Encode/Create dummies for categorical attributes and return the data and labels

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame: Data
        pd.DataFrame: Labels
    """

    categorical_columns_encode = [
        "Reason for Delay",
        "Departure Station",
        "Arrival Destination",
    ]
    labelEncoder = [LabelEncoder()] * len(categorical_columns_encode)
    labelEncoder_y = LabelEncoder()

    # Encode categorical columns containing many categories
    for i, col in enumerate(categorical_columns_encode):
        if col in df.columns:
            df[col] = labelEncoder[i].fit_transform(df[col])
    # Encode dependent variable
    df["Refund Request"] = labelEncoder_y.fit_transform(df["Refund Request"])

    # To create dummy variable for journey.status with delayed and cancelled columns
    if "Journey Status" in df.columns:
        df["Journey Status"] = pd.Categorical(
            df["Journey Status"],
            categories=["On Time", "Delayed", "Cancelled"],
            ordered=True,
        )

    # Define the independent variables
    categorical_columns_encode = [
        "Reason for Delay",
        "Departure Station",
        "Arrival Destination",
    ]
    numerical_columns = ["Price", "DelayInMinutes"]
    dummy_columns = [
        "Payment Method",
        "Railcard",
        "Ticket Type",
        "Ticket Class",
        "Journey Status",
    ]

    # Select only the columns that exist in the DataFrame
    X_numerical = df[[col for col in numerical_columns if col in df.columns]]

    X_categorical = df[[col for col in categorical_columns_encode if col in df.columns]]

    X_dummies = pd.get_dummies(
        df[[col for col in dummy_columns if col in df.columns]],
        drop_first=True,
    ).astype(int)

    # Concatenate all selected columns
    X = pd.concat([X_dummies, X_categorical, X_numerical], axis=1)

    # Define the dependent variable
    y = df["Refund Request"]

    return X, y


def save_csv_files(data: pd.DataFrame, labels: pd.DataFrame):
    """
    Save the data and labels csv files in the 'datasets' directory

    Args:
        data (pd.DataFrame): Attributes in the model
        labels (pd.DataFrame): Prediction variable

    Returns:
    File paths
    """
    base_path = Path().resolve()

    data_file_path = os.path.join(base_path, "datasets", "data.csv")
    labels_file_path = os.path.join(base_path, "datasets", "labels.csv")

    data.to_csv(data_file_path, index=False)
    labels.to_csv(labels_file_path, index=False)

    return data_file_path, labels_file_path


def split_dataset(data_file_path, labels_file_path, test_size, random_state=42):
    """
    Split the dataset into train and test based on the test size

    Args:
        data_file_path (string): Path to the data dataset
        labels_file_path (string): Path to the labels dataset
        test_size (float): Proportion of data to be allocated for the test set
        random_state (int, optional): Defaults to 42.

    Returns:
        int: length of the 4 datasets
    """
    X = pd.read_csv(data_file_path)
    y = pd.read_csv(labels_file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    base_path = Path().resolve()

    train_path = os.path.join(base_path, "datasets", "train")
    test_path = os.path.join(base_path, "datasets", "test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for df, file_path, csv_name in zip(
        [X_train, X_test, y_train, y_test],
        [train_path, test_path] * 2,
        ["data.csv", "data.csv", "labels.csv", "labels.csv"],
    ):
        csv_path = os.path.join(file_path, csv_name)
        df.to_csv(csv_path, index=False)

    return len(X_train), len(X_test), len(y_train), len(y_test)


def save_model(model):
    """
    Saves the trained logistic regression model and its summary statistics

    Args:
        model (sm.Logit): Trained model

    Returns:
        object: Summary of trained model
    """
    base_path = Path().resolve()
    model_path = os.path.join(base_path, "log_regression")
    os.makedirs(model_path, exist_ok=True)

    # save results as csv
    summary = model.summary2()
    summary_df = summary.tables[1]
    summary_df.to_csv(os.path.join(model_path, "model_metrics.csv"), index=True)

    # Remove large data arrays to reduce size
    model._results.remove_data()

    # Save the model using pickle
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    return summary


def reduce_columns(columns):
    """_summary_

    Args:
        columns (_type_): _description_
    """
    base_path = Path().resolve()
    data = pd.read_csv(
        os.path.join(base_path, "datasets", "TrainRidesCleaned.csv"), delimiter=";"
    )
    columns.append("Refund Request")
    data_reduced = data[columns]
    data_reduced.to_csv(
        os.path.join(base_path, "datasets", "TrainRidesReduced.csv"), index=False
    )

    return data_reduced
