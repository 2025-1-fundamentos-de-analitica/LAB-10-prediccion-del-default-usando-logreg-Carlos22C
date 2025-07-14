import os
import gzip
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)

def load_data(csv_file):
    return pd.read_csv(csv_file, compression="zip")

def data_clean(data):
    df = data.copy()
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda v: 4 if v > 4 else v)
    return df

def split_data(train_df, test_df):
    X_train = train_df.drop(columns="default")
    y_train = train_df["default"]
    X_test = test_df.drop(columns="default")
    y_test = test_df["default"]
    return X_train, y_train, X_test, y_test

def create_pipeline():
    categorical = ["EDUCATION", "SEX", "MARRIAGE"]
    preprocessing = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(), categorical)],
        remainder=MinMaxScaler()
    )
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("selector", SelectKBest(score_func=f_regression, k=10)),
        ("classifier", LogisticRegression(solver="liblinear", random_state=42))
    ])
    return pipeline

def make_grid_search(pipeline):
    param_grid = {
        "selector__k": list(range(1, 11)),
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear"],
        "classifier__max_iter": [100, 200]
    }
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        verbose=2
    )

def save_model(estimator, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)

def check_estimator(estimator, X, y, dataset_name):
    y_pred = estimator.predict(X)
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": round(precision_score(y, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "f1_score": round(f1_score(y, y_pred), 4)
    }, y_pred, y

def c_matrix(y_true, y_pred, dataset):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1])
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1])
        }
    }

def main():
    os.makedirs("files/output", exist_ok=True)

    train_df = data_clean(load_data("files/input/train_data.csv.zip"))
    test_df = data_clean(load_data("files/input/test_data.csv.zip"))

    X_train, y_train, X_test, y_test = split_data(train_df, test_df)

    pipeline = create_pipeline()
    search = make_grid_search(pipeline)
    best_model = search.fit(X_train, y_train)

    metrics_train, y_pred_train, y_train = check_estimator(best_model, X_train, y_train, "train")
    metrics_test, y_pred_test, y_test = check_estimator(best_model, X_test, y_test, "test")

    matrix_train = c_matrix(y_train, y_pred_train, "train")
    matrix_test = c_matrix(y_test, y_pred_test, "test")

    with open("files/output/metrics.json", "w") as f:
        for record in [metrics_train, metrics_test, matrix_train, matrix_test]:
            f.write(json.dumps(record) + "\n")

    save_model(best_model, "files/models/model.pkl.gz")

main()
