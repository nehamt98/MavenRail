import pickle
import os
import pandas as pd

from src.schemas import TransactionData, InferenceData
from utils.main_utils import process_time, transformation


def data_process(transaction_data: TransactionData):
    """_summary_

    Args:
        transaction_data (TransactionData): _description_
    """
    # Create DataFrame from list of TransactionData objects
    df = transaction_data.to_dataframe()

    processed_df = process_time(df)

    data = transformation(processed_df, load_encoder_flag=True)

    prediction = perform_inference(data)

    result = {"Refund Request": prediction}

    return result


def perform_inference(data: pd.DataFrame):
    """_summary_

    Args:
        data (pd.DataFrame): _description_
    """
    with open(os.path.join("log_regression", "model.pkl"), "rb") as f:
        loaded_model = pickle.load(f)

    # Convert DataFrame to a 1D NumPy array of shape (15,)
    data_array = data.values.flatten()

    # Ensure the correct shape
    if data_array.shape[0] != 15:
        raise ValueError(f"Expected input shape (15,), but got {data_array.shape}")

    # Make Prediction
    pred_prob = loaded_model.predict(data_array.reshape(1, -1))  # Reshape to (1, 15)

    pred = (pred_prob >= 0.5).astype(int)

    return "Yes" if pred[0] == 1 else "No"
