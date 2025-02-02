import sys

import os
import pickle

import pandas as pd
from pathlib import Path


import statsmodels.api as sm


def main():

    base_path = Path().resolve()
    y_train = pd.read_csv(os.path.join(base_path, "datasets", "train", "labels.csv"))
    X_train = pd.read_csv(os.path.join(base_path, "datasets", "train", "data.csv"))

    model = sm.Logit(y_train, X_train).fit()

    # Remove large data arrays to reduce size
    model._results.remove_data()

    # Save the model using pickle
    model_path = os.path.join(base_path, "log_regression")
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # save results as csv
    summary_df = model.summary2()

    summary_df.to_csv(os.path.join(model_path, "model_metrics.csv"), index=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
