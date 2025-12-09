import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(x_test_path, y_test_path, model_path, report_out, cm_out):
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found at {model_path}. Run train first.")

    clf = joblib.load(model_path)

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).iloc[:, 0]

    preds = clf.predict(X_test)

    report = classification_report(y_test, preds)
    with open(report_out, "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    d = os.path.dirname(cm_out)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    plt.savefig(cm_out, bbox_inches="tight")
    plt.close()

    print(report)
    print(f"Saved classification report to {report_out}")
    print(f"Saved confusion matrix image to {cm_out}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--x-test", default="data/X_test.csv")
    parser.add_argument("--y-test", default="data/y_test.csv")
    parser.add_argument("--model", default="models/random_forest.pkl")
    parser.add_argument("--report-out", default="results/classification_report.txt")
    parser.add_argument("--cm-out", default="results/confusion_matrix.png")
    args = parser.parse_args()

    evaluate(args.x_test, args.y_test, args.model, args.report_out, args.cm_out)


if __name__ == "__main__":
    main()
