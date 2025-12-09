





import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def find_target_column(df):
    candidates = ["target", "label", "y", "default"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def preprocess(in_path, processed_out, x_train_out, y_train_out, x_test_out, y_test_out, test_size=0.2, random_state=42):
    df = pd.read_csv(in_path)
    target = find_target_column(df)
    if target is None:
        raise SystemExit("No target column found. Expected one of: target,label,y,default")

    # Basic cleaning
    df = df.dropna()

    y = df[target].reset_index(drop=True)
    X = df.drop(columns=[target])

    # Convert categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train/test split
    stratify = y if len(y.unique()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Ensure output directories exist
    for p in [processed_out, x_train_out, y_train_out, x_test_out, y_test_out]:
        d = os.path.dirname(p)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # Save files
    processed_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    processed_df.to_csv(processed_out, index=False)
    X_train.to_csv(x_train_out, index=False)
    y_train.to_csv(y_train_out, index=False)
    X_test.to_csv(x_test_out, index=False)
    y_test.to_csv(y_test_out, index=False)

    print(f"Saved processed dataset to {processed_out}")
    print(f"Saved X_train to {x_train_out}, y_train to {y_train_out}")
    print(f"Saved X_test to {x_test_out}, y_test to {y_test_out}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset and split train/test")
    parser.add_argument("--input", default="data/dataset.csv")
    parser.add_argument("--processed", default="data/processed.csv")
    parser.add_argument("--x-train", default="data/X_train.csv")
    parser.add_argument("--y-train", default="data/y_train.csv")
    parser.add_argument("--x-test", default="data/X_test.csv")
    parser.add_argument("--y-test", default="data/y_test.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    preprocess(
        args.input,
        args.processed,
        args.x_train,
        args.y_train,
        args.x_test,
        args.y_test,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()


