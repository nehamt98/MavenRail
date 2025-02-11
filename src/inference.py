import sys

import os
import pickle

import pandas as pd
from pathlib import Path

from sklearn.metrics import confusion_matrix, accuracy_score

from utils.main_utils import load_model


def main():

    base_path = Path().resolve()

    # Load datasets
    y_train = pd.read_csv(os.path.join(base_path, "datasets", "train", "labels.csv"))
    X_train = pd.read_csv(os.path.join(base_path, "datasets", "train", "data.csv"))
    X_test = pd.read_csv(os.path.join(base_path, "datasets", "test", "data.csv"))
    y_test = pd.read_csv(os.path.join(base_path, "datasets", "test", "labels.csv"))

    # Load the model
    loaded_model = load_model()["model"]

    # Predict for training and test data
    y_pred_prob_train = loaded_model.predict(X_train)
    y_pred_train = (y_pred_prob_train >= 0.5).astype(int)
    y_pred_prob_test = loaded_model.predict(X_test)
    y_pred_test = (y_pred_prob_test >= 0.5).astype(int)

    # Calculate the score for training and test data
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f"Accuracy for training data: {accuracy_train:.2f}")
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy for test data: {accuracy_test:.2f}")

    # Compute confusion matrix for training and test data
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    print("\nConfusion Matrix for training data:")
    print(conf_matrix_train)
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix for test data:")
    print(conf_matrix_test)

    sys.exit(0)


if __name__ == "__main__":
    main()
