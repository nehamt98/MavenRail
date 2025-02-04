import argparse
import sys

from utils.main_utils import (
    load_dataset,
    clean_dataset,
    process_time,
    reduce_columns,
    transformation,
    save_csv_files,
    split_dataset,
)


def main(args):
    print(args)

    df = load_dataset(args.filename)

    cleaned_df = clean_dataset(df)

    processed_df = process_time(cleaned_df)

    if args.columns:
        processed_df = reduce_columns(args.columns)

    data, labels = transformation(processed_df, save_encoder_flag=True)

    data_file_path, label_file_path = save_csv_files(data, labels)

    X_train, X_test, y_train, y_test = split_dataset(
        data_file_path, label_file_path, args.test_size, random_state=args.random_state
    )

    print(
        f"lengths - Xtrain: {X_train}; Xtest: {X_test}; Ytrain: {y_train}; Ytest: {y_test};"
    )
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="basic desc")

    parser.add_argument("--filename", type=str, required=True)

    parser.add_argument("--test_size", type=float, required=True)

    parser.add_argument("--random_state", type=int, required=False)

    parser.add_argument("--columns", type=str, nargs="+", required=False)

    args = parser.parse_args()
    main(args)
