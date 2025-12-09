import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


def train(x_train_path, y_train_path, model_out, n_estimators=100, random_state=42):
    if os.path.exists(x_train_path) and os.path.exists(y_train_path):
        X = pd.read_csv(x_train_path)
        y = pd.read_csv(y_train_path).iloc[:, 0]
    else:
        if not os.path.exists("data/processed.csv"):
            raise SystemExit("No training data found. Run preprocess first to create data/processed.csv or provide X_train/y_train paths.")
        df = pd.read_csv("data/processed.csv")
        if "target" in df.columns:
            y = df["target"]
            X = df.drop(columns=["target"])
        else:
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]

    d = os.path.dirname(model_out)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X, y)
    joblib.dump(clf, model_out)
    print(f"Saved model to {model_out}")


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest model")
    parser.add_argument("--x-train", default="data/X_train.csv")
    parser.add_argument("--y-train", default="data/y_train.csv")
    parser.add_argument("--model-out", default="models/random_forest.pkl")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train(args.x_train, args.y_train, args.model_out, n_estimators=args.n_estimators, random_state=args.random_state)


if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("../data/processed.csv")

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

with open("../models/random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

X_test.to_csv("../data/X_test.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

print("Model eğitim tamam.")
