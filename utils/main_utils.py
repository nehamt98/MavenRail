import os
import pandas as pd
from pathlib import Path
import pickle

from loguru import logger
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# File paths for saving encoders
OHE_ENCODER_FILE = "onehot_encoder.pkl"
LABEL_ENCODER_FILE = "label_encoders.pkl"


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
    if all(col in df.columns for col in ["Actual Arrival Time", "Arrival Time"]):
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


def save_encoders(
    onehot_encoder,
    label_encoders,
    ohe_filename=OHE_ENCODER_FILE,
    le_filename=LABEL_ENCODER_FILE,
):
    """Save OneHotEncoder and LabelEncoders using pickle."""
    base_path = Path().resolve()
    model_path = os.path.join(base_path, "log_regression")
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, ohe_filename), "wb") as f:
        pickle.dump(onehot_encoder, f)
    with open(os.path.join(model_path, le_filename), "wb") as f:
        pickle.dump(label_encoders, f)


def load_encoders(ohe_filename=OHE_ENCODER_FILE, le_filename=LABEL_ENCODER_FILE):
    """Load OneHotEncoder and LabelEncoders from pickle files."""
    base_path = Path().resolve()
    model_path = os.path.join(base_path, "log_regression")
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, ohe_filename), "rb") as f:
        onehot_encoder = pickle.load(f)
    with open(os.path.join(model_path, le_filename), "rb") as f:
        label_encoders = pickle.load(f)
    return onehot_encoder, label_encoders


def transformation(df: pd.DataFrame, save_encoder_flag=False, load_encoder_flag=False):
    """
    Encode categorical attributes using OneHotEncoder and LabelEncoder.
    Supports both training (saving encoders) and inference (loading encoders).

    Args:
        df (pd.DataFrame): Input DataFrame
        save_encoder_flag (bool): Saves encoders if True
        load_encoder_flag (bool): Loads pre-trained encoders if True

    Returns:
        X (pd.DataFrame): Encoded feature data
        y (pd.Series or None): Labels if available, else None
    """

    # Columns for encoding
    categorical_columns_encode = [
        "Reason for Delay",
        "Departure Station",
        "Arrival Destination",
    ]
    numerical_columns = ["Price", "DelayInMinutes"]
    onehot_columns = [
        "Payment Method",
        "Railcard",
        "Ticket Type",
        "Ticket Class",
        "Journey Status",
    ]

    if load_encoder_flag:
        encoder, label_encoders = load_encoders()
    else:
        label_encoders = {col: LabelEncoder() for col in categorical_columns_encode}
        encoder = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )

    # Label Encode categorical columns
    for col in categorical_columns_encode:
        if col in df.columns:
            if load_encoder_flag:
                df[col] = label_encoders[col].transform(df[col])
            else:
                df[col] = label_encoders[col].fit_transform(df[col])

    # Encode dependent variable
    if "Refund Request" in df.columns:
        df["Refund Request"] = df["Refund Request"].map({"No": 0, "Yes": 1})

    # OneHotEncode categorical features
    existing_onehot = [col for col in onehot_columns if col in df.columns]

    if existing_onehot:
        if load_encoder_flag:
            encoded_array = encoder.transform(df[existing_onehot])
        else:
            encoded_array = encoder.fit_transform(df[existing_onehot])

        # Convert to DataFrame
        encoded_df = pd.DataFrame(
            encoded_array, columns=encoder.get_feature_names_out(existing_onehot)
        )

    # Save encoders after training
    if save_encoder_flag:
        save_encoders(encoder, label_encoders)

    # Select numerical columns
    existing_numerical = [col for col in numerical_columns if col in df.columns]
    X_numerical = df[existing_numerical] if existing_numerical else pd.DataFrame()

    # Select label-encoded categorical columns
    existing_categorical = [
        col for col in categorical_columns_encode if col in df.columns
    ]
    X_categorical = df[existing_categorical] if existing_categorical else pd.DataFrame()

    # Combine feature DataFrames
    X = pd.concat([X_numerical, X_categorical, encoded_df], axis=1)

    # Define labels if available
    y = df["Refund Request"] if "Refund Request" in df.columns else None

    return (X, y) if y is not None else X


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
