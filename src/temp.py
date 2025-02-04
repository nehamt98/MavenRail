import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# File paths for saving encoders
OHE_ENCODER_FILE = "onehot_encoder.pkl"
LABEL_ENCODER_FILE = "label_encoders.pkl"


def save_encoders(
    onehot_encoder,
    label_encoders,
    ohe_filename=OHE_ENCODER_FILE,
    le_filename=LABEL_ENCODER_FILE,
):
    """Save OneHotEncoder and LabelEncoders using pickle."""
    with open(ohe_filename, "wb") as f:
        pickle.dump(onehot_encoder, f)
    with open(le_filename, "wb") as f:
        pickle.dump(label_encoders, f)


def load_encoders(ohe_filename=OHE_ENCODER_FILE, le_filename=LABEL_ENCODER_FILE):
    """Load OneHotEncoder and LabelEncoders from pickle files."""
    with open(ohe_filename, "rb") as f:
        onehot_encoder = pickle.load(f)
    with open(le_filename, "rb") as f:
        label_encoders = pickle.load(f)
    return onehot_encoder, label_encoders


def transformation(df: pd.DataFrame, save_encoder_flag=False):
    """
    Encode categorical attributes using OneHotEncoder and LabelEncoder.

    Args:
        df (pd.DataFrame): Input DataFrame
        save_encoder_flag (bool): If True, saves the encoders for future use.

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

    # Initialize LabelEncoders for categorical columns
    label_encoders = {col: LabelEncoder() for col in categorical_columns_encode}

    # Encode categorical columns (Label Encoding)
    for col in categorical_columns_encode:
        if col in df.columns:
            df[col] = label_encoders[col].fit_transform(df[col])

    # Encode dependent variable
    if "Refund Request" in df.columns:
        df["Refund Request"] = df["Refund Request"].map({"No": 0, "Yes": 1})

    # OneHotEncode categorical features
    existing_onehot = [col for col in onehot_columns if col in df.columns]

    if existing_onehot:
        encoder = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )
        encoded_array = encoder.fit_transform(df[existing_onehot])

        # Convert to DataFrame
        encoded_df = pd.DataFrame(
            encoded_array, columns=encoder.get_feature_names_out(existing_onehot)
        )

        # Save both encoders for future use
        if save_encoder_flag:
            save_encoders(encoder, label_encoders)

        # Drop original categorical columns and add the encoded ones
        df = df.drop(columns=existing_onehot).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)

    # Select numerical columns
    existing_numerical = [col for col in numerical_columns if col in df.columns]
    X_numerical = df[existing_numerical] if existing_numerical else pd.DataFrame()

    # Select label-encoded categorical columns
    existing_categorical = [
        col for col in categorical_columns_encode if col in df.columns
    ]
    X_categorical = df[existing_categorical] if existing_categorical else pd.DataFrame()

    # Combine feature DataFrames
    X = pd.concat(
        [
            X_numerical,
            X_categorical,
            df.drop(columns=["Refund Request"], errors="ignore"),
        ],
        axis=1,
    )

    # Define labels if available
    y = df["Refund Request"] if "Refund Request" in df.columns else None

    return (X, y) if y is not None else (X, label_encoders)
