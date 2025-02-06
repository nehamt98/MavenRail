import pandas as pd

from src.schemas import TransactionData
from utils.main_utils import clean_dataset, process_time, transformation, load_model


def data_process(transaction_data: TransactionData):
    """Preprocessing and inference of the API request

    Args:
        transaction_data (TransactionData): Features returned from API

    Returns: result (Dictionary): Predicted refund request
    """
    # Create DataFrame from list of TransactionData objects
    df = transaction_data.to_dataframe()

    # Process the data
    cleaned_df = clean_dataset(df)

    processed_df = process_time(cleaned_df)

    data = transformation(processed_df, load_encoder_flag=True)

    # Make the prediction
    prediction = perform_inference(data)

    result = {"Refund Request": prediction}

    return result


def perform_inference(data: pd.DataFrame):
    """Makes the prediction

    Args:
        data (pd.DataFrame): Cleaned and transformed dataframe

    Returns: String: Yes or No based on the prediction
    """
    loaded_model = load_model()["model"]

    # Make Prediction
    pred_prob = loaded_model.predict(data.values)

    # Assign 0 or 1 based on probability
    pred = (pred_prob >= 0.5).astype(int)

    return "Yes" if pred[0] == 1 else "No"
