import os
import pandas as pd
from pathlib import Path

from loguru import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_dataset(filename: str) -> pd.DataFrame:
    """_summary_

    Args:
        filename (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    base_path = Path().resolve()

    try:
        df = pd.read_csv(os.path.join(base_path, "datasets", filename), delimiter=";")
    except Exception as e:
        logger.error(e)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df["Railcard"] = df["Railcard"].fillna("None")
    df["Reason for Delay"] = df["Reason for Delay"].fillna("Not Delayed")
    df["Date of Journey"] = pd.to_datetime(df["Date of Journey"], format="%d/%m/%Y")

    return df


def process_time(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
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
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    maven_data_ML = df.copy()

    categorical_columns_encode = [
        "Reason for Delay",
        "Departure Station",
        "Arrival Destination",
    ]
    labelEncoder = [LabelEncoder()] * len(categorical_columns_encode)
    labelEncoder_y = LabelEncoder()

    # Encode categorical columns with many categories
    for i, col in enumerate(categorical_columns_encode):
        maven_data_ML[col] = labelEncoder[i].fit_transform(maven_data_ML[col])
    # Encode dependent variable
    maven_data_ML["Refund Request"] = labelEncoder_y.fit_transform(
        maven_data_ML["Refund Request"]
    )

    # To create dummy variable for journey.status with delayed and cancelled columns
    maven_data_ML["Journey Status"] = pd.Categorical(
        maven_data_ML["Journey Status"],
        categories=["On Time", "Delayed", "Cancelled"],
        ordered=True,
    )

    # Define the independent variables
    X_numerical = maven_data_ML[["Price", "DelayInMinutes"]]
    X_categorical = maven_data_ML[categorical_columns_encode]
    X_dummies = pd.get_dummies(
        maven_data_ML[
            [
                "Payment Method",
                "Railcard",
                "Ticket Type",
                "Ticket Class",
                "Journey Status",
            ]
        ],
        drop_first=True,
    ).astype(int)
    X = pd.concat([X_dummies, X_categorical, X_numerical], axis=1)
    # Define the dependent variable
    y = maven_data_ML["Refund Request"]

    return X, y


def save_csv_files(data: pd.DataFrame, labels: pd.DataFrame):
    """_summary_

    Args:
        data (_type_): _description_
        labels (_type_): _description_
    """
    base_path = Path().resolve()

    data_file_path = os.path.join(base_path, "datasets", "data.csv")
    labels_file_path = os.path.join(base_path, "datasets", "labels.csv")

    data.to_csv(data_file_path)
    labels.to_csv(labels_file_path)

    return data_file_path, labels_file_path


def split_dataset(data_file_path, labels_file_path, test_size, random_state=42):
    """_summary_

    Args:
        data_file_path (_type_): _description_
        labels_file_path (_type_): _description_
        test_size (_type_): _description_
        random_state (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
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
        df.to_csv(csv_path)

    return len(X_train), len(X_test), len(y_train), len(y_test)
