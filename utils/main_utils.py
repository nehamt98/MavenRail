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
    Imputation and data formatting are carried out for the columns - Railcard and Reason for Delay

    Args:
        df (pd.DataFrame): The loaded dataset

    Returns:
        pd.DataFrame: The cleaned dataset
    """
    df["Railcard"] = df["Railcard"].fillna("None")
    df["Reason for Delay"] = df["Reason for Delay"].fillna("Not Delayed")
    df["Reason for Delay"] = df["Reason for Delay"].replace("NA", "Not Delayed")

    return df


def process_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    A new column 'DelayInMinutes' is calculated based on the Actual and Scheduled arrival times

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
    """
    Save OneHotEncoder and LabelEncoders using pickle.

    Args:
        onehot_encoder (<class sklearn.preprocessing._encoders.OneHotEncoder>): One hot encoder fit on training data
        label_encoders (<class 'sklearn.preprocessing._label.LabelEncoder'>): Label encoder fit on training data
        ohe_filename (string, optional): File path to OHE. Defaults to OHE_ENCODER_FILE.
        le_filename (string, optional): File path to LE. Defaults to LABEL_ENCODER_FILE.
    """
    base_path = Path().resolve()
    model_path = os.path.join(base_path, "log_regression")
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, ohe_filename), "wb") as f:
        pickle.dump(onehot_encoder, f)
    with open(os.path.join(model_path, le_filename), "wb") as f:
        pickle.dump(label_encoders, f)


def load_encoders(ohe_filename=OHE_ENCODER_FILE, le_filename=LABEL_ENCODER_FILE):
    """
    Load OneHotEncoder and LabelEncoders from pickle files.

    Args:
        ohe_filename (string, optional): File path to OHE. Defaults to OHE_ENCODER_FILE.
        le_filename (string, optional): File path to LE. Defaults to LABEL_ENCODER_FILE.

    Returns:
        onehot_encoder (<class sklearn.preprocessing._encoders.OneHotEncoder>): One hot encoder fit on training data
        label_encoders (<class 'sklearn.preprocessing._label.LabelEncoder'>): Label encoder fit on training data
    """
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
        save_encoder_flag (bool, optional): Saves encoders if True
        load_encoder_flag (bool, optional): Loads pre-trained encoders if True

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

    # Load or initialise encoders based on flags
    if load_encoder_flag:
        encoder, label_encoders = load_encoders()

    else:
        label_encoders = {}
        encoder = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )

    # Label Encode categorical columns
    for col in categorical_columns_encode:
        if col in df.columns:
            if load_encoder_flag:
                if col in label_encoders.keys():
                    df[col] = label_encoders[col].transform(df[col])
            else:
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
                label_encoders[col] = label_encoder

    # Encode dependent variable
    if "Refund Request" in df.columns:
        df["Refund Request"] = df["Refund Request"].map({"No": 0, "Yes": 1})

    # OneHotEncode categorical features
    existing_onehot = [col for col in onehot_columns if col in df.columns]

    if existing_onehot:
        if load_encoder_flag:
            valid_columns = [
                col for col in existing_onehot if col in encoder.feature_names_in_
            ]
            if valid_columns:  # Proceed only if there are valid columns
                encoded_array = encoder.transform(df[valid_columns])
        else:
            encoded_array = encoder.fit_transform(df[existing_onehot])
            valid_columns = existing_onehot

        # Convert to DataFrame
        encoded_df = pd.DataFrame(
            encoded_array, columns=encoder.get_feature_names_out(valid_columns)
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

    if load_encoder_flag:
        # Drop columns that are not in the saved encoders. Mainly done for the numerical columns
        feature_names = load_model()["feature_names"]
        X = X[feature_names]

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

    # Save feature names
    feature_names = model.params.keys()

    # Remove large data arrays to reduce size
    model._results.remove_data()

    # Save the model using pickle
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "feature_names": list(feature_names)}, f)
    return summary


def reduce_columns(processed_df, columns):
    """
    Select columns that are passed to the function for model training

    Args:
        processed_df (pd.DataFrame): Complete dataframe
        columns (List): Columns to be retained

    Returns:
        pd.DataFrame: Reduced dataframe
    """
    base_path = Path().resolve()

    columns.append("Refund Request")
    data_reduced = processed_df[columns]
    data_reduced.to_csv(
        os.path.join(base_path, "datasets", "TrainRidesReduced.csv"), index=False
    )

    return data_reduced


def load_model(model_name="model.pkl"):
    """
    Load the saved model

    Args:
        model_name (str, optional): Defaults to "model.pkl".

    Returns:
        model
    """
    base_path = Path().resolve()
    model_path = os.path.join(base_path, "log_regression")
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(base_path, model_path, "model.pkl"), "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model
