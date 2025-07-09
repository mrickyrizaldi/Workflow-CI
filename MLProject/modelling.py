import pandas as pd
import mlflow
import mlflow.sklearn
import time
import sys
import os
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Ambil parameter dari sys.argv (bisa dari CLI atau pakai default)
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    min_samples_split = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    min_samples_leaf = int(sys.argv[4]) if len(sys.argv) > 4 else 9

    # Dataset Path (Bisa lewat sys.argv atau default)
    dataset_dir = sys.argv[5] if len(sys.argv) > 5 else 'preprocessed_data_auto_20250701_145341'
    train_path = os.path.join(dataset_dir, 'X_train.csv')
    test_path = os.path.join(dataset_dir, 'X_test.csv')
    y_train_path = os.path.join(dataset_dir, 'y_train.csv')
    y_test_path = os.path.join(dataset_dir, 'y_test.csv')

    # Load Dataset
    try:
        X_train = pd.read_csv(train_path)
        X_test = pd.read_csv(test_path)
        y_train = pd.read_csv(y_train_path).values.ravel()
        y_test = pd.read_csv(y_test_path).values.ravel()

        input_example = X_train.iloc[:5]
        print("Dataset berhasil dimuat")

    except Exception as e:
        print(f"Gagal memuat dataset: {e}")
        sys.exit(1)

    # Training Random Forest
    print("\nTraining Random Forest Classifier")
    try:
        with mlflow.start_run(run_name="RandomForestClassifier"):
            mlflow.autolog()

            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )

            start = time.time()
            rf_model.fit(X_train, y_train)
            end = time.time()

            y_pred_rf = rf_model.predict(X_test)

            # Manual additional logging
            training_acc = rf_model.score(X_train, y_train)
            acc_rf = accuracy_score(y_test, y_pred_rf)
            f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
            precision = precision_score(y_test, y_pred_rf, average="weighted")
            recall = recall_score(y_test, y_pred_rf, average="weighted")
            loss = log_loss(y_test, rf_model.predict_proba(X_test))

            mlflow.log_metric("training_accuracy", training_acc)
            mlflow.log_metric("test_accuracy", acc_rf)
            mlflow.log_metric("f1_score", f1_rf)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("log_loss", loss)
            mlflow.log_metric("training_time_sec", end - start)

            mlflow.sklearn.log_model(
                sk_model=rf_model,
                artifact_path="random_forest_model",
                input_example=input_example
            )

            print(f"Random Forest - Test Accuracy: {acc_rf:.4f}, F1-Score: {f1_rf:.4f}, Log Loss: {loss:.4f}")

    except Exception as e:
        print(f"Gagal training Random Forest: {e}")
