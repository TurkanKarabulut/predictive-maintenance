import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from prepare_data import FEATURES, TARGET


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 42


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_pipeline(estimator):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                FEATURES[1:],
            ),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                ["Type"],
            ),
        ]
    )
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", estimator),
    ])


def evaluate(pipeline, X, y):
    start = perf_counter()
    probabilities = pipeline.predict_proba(X)[:, 1]
    elapsed = perf_counter() - start
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "batch_inference_ms": elapsed * 1000,
        "batch_size": len(X),
    }
    return metrics, predictions, probabilities


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    csv_path = ROOT / "data" / "raw" / "ai4i2020.csv"
    df = pd.read_csv(csv_path)
    X = df[FEATURES]
    y = df[TARGET]

    # Once ayrim; sonra pipeline egitimi.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=SEED,
    )

    split_indices = {
        "train": X_train.index.tolist(),
        "validation": X_val.index.tolist(),
        "test": X_test.index.tolist(),
    }
    save_json(REPORTS / "split_indices.json", split_indices)

    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    fitted = {}
    rows = []

    for name, estimator in candidates.items():
        print(f"Egitiliyor: {name}", flush=True)
        pipeline = build_pipeline(estimator)

        start = perf_counter()
        pipeline.fit(X_train, y_train)
        training_seconds = perf_counter() - start

        metrics, _, _ = evaluate(pipeline, X_val, y_val)
        rows.append({
            "model": name,
            "training_seconds": training_seconds,
            **metrics,
        })
        fitted[name] = pipeline

    comparison = pd.DataFrame(rows).sort_values(
        by=["f1", "recall", "average_precision", "model"],
        ascending=[False, False, False, True],
    )
    comparison.to_csv(
        REPORTS / "validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Secim yalnizca validation sonuclarina gore yapilir.
    best_name = str(comparison.iloc[0]["model"])
    best_pipeline = fitted[best_name]

    # Test oncesi yeniden egitim yapilmiyor:
    # kaydedilen model, validation ile secilen ayni modeldir.
    test_metrics, predictions, probabilities = evaluate(
        best_pipeline, X_test, y_test
    )

    version = datetime.now(timezone.utc).strftime(
        "v%Y%m%dT%H%M%S%fZ"
    )
    artifact_name = f"model_{version}.joblib"
    joblib.dump(best_pipeline, MODELS / artifact_name)

    metadata = {
        "model_name": best_name,
        "model_version": version,
        "artifact": artifact_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "decision_threshold": 0.5,
        "selection_rule": (
            "Highest validation F1; ties: recall, "
            "average precision, model name"
        ),
        "features": FEATURES,
        "target": TARGET,
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "split_class_counts": {
            name: {
                str(k): int(v)
                for k, v in y.loc[indices].value_counts().items()
            }
            for name, indices in split_indices.items()
        },
        "data_sha256": sha256(csv_path.read_bytes()).hexdigest(),
        "estimator_parameters": (
            best_pipeline.named_steps["model"].get_params()
        ),
        "test_metrics": test_metrics,
    }
    save_json(MODELS / "metadata.json", metadata)
    save_json(REPORTS / "test_metrics.json", test_metrics)

    test_output = X_test.copy()
    test_output.insert(0, "source_row_index", X_test.index)
    test_output["actual"] = y_test.to_numpy()
    test_output["prediction"] = predictions
    test_output["probability"] = probabilities
    test_output["error_type"] = [
        "FN" if actual == 1 and predicted == 0
        else "FP" if actual == 0 and predicted == 1
        else "correct"
        for actual, predicted in zip(y_test, predictions)
    ]
    test_output.to_csv(
        REPORTS / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["Normal", "Failure"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title(f"Test Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(
        REPORTS / "test_confusion_matrix.png",
        dpi=160,
    )
    plt.close()

    print("\nDOGRULAMA SONUCLARI:")
    print(comparison[
        ["model", "precision", "recall", "f1", "roc_auc"]
    ].to_string(index=False))

    print(f"\nSecilen model: {best_name}")
    print("\nTEST SONUCLARI:")
    print(json.dumps(test_metrics, indent=2))
    print("\nModel ve rapor dosyalari kaydedildi.")


if __name__ == "__main__":
    main()