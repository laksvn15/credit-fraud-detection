import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# ==== Load dataset ====
data = pd.read_csv("data/creditcard.csv")
features = data.drop("Class", axis=1)
labels = data["Class"]

# ==== Feature normalization ====
scaler = StandardScaler()
features["Amount"] = scaler.fit_transform(features["Amount"].values.reshape(-1, 1))
features["Time"] = scaler.fit_transform(features["Time"].values.reshape(-1, 1))

# ==== Split data ====
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

# ==== Handle class imbalance with SMOTE ====
oversampler = SMOTE(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

# ==== Define classification models ====
positive_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()

models_to_train = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "XGBoost Classifier": XGBClassifier(
        scale_pos_weight=positive_weight_ratio,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
}

# ==== Create output directory ====
os.makedirs("outputs", exist_ok=True)
output_file = "outputs/classification_report.txt"

# ==== Training and Evaluation ====
with open(output_file, "w") as log_file:
    for model_name, clf in models_to_train.items():
        print(f"\n--- {model_name} ---")
        log_file.write(f"\n--- {model_name} ---\n")

        start_time = time.time()
        clf.fit(X_resampled, y_resampled)
        elapsed = time.time() - start_time
        print(f"Training time: {elapsed:.2f}s")

        y_score = clf.predict_proba(X_test)[:, 1]

        # Default threshold
        y_pred_05 = (y_score > 0.5).astype(int)
        auc_score = roc_auc_score(y_test, y_score)
        conf_matrix = confusion_matrix(y_test, y_pred_05)
        report = classification_report(y_test, y_pred_05, digits=4)

        print(f"ROC-AUC Score: {auc_score:.4f}")
        print("Classification Report (Threshold=0.5):\n", report)

        log_file.write(f"ROC-AUC Score: {auc_score:.4f}\n")
        log_file.write("Classification Report (Threshold=0.5):\n")
        log_file.write(report + "\n")
        log_file.write("Confusion Matrix:\n")
        log_file.write(np.array2string(conf_matrix) + "\n\n")

        # Custom threshold
        threshold = 0.8
        y_pred_08 = (y_score > threshold).astype(int)
        conf_matrix_08 = confusion_matrix(y_test, y_pred_08)
        report_08 = classification_report(y_test, y_pred_08, digits=4)

        print(f"Classification Report (Threshold={threshold}):\n", report_08)
        log_file.write(f"Classification Report (Threshold={threshold}):\n")
        log_file.write(report_08 + "\n")
        log_file.write("Confusion Matrix:\n")
        log_file.write(np.array2string(conf_matrix_08) + "\n\n")


# ==== Precision-Recall Curve Plotting ====
def generate_precision_recall_plot(models_dict, X_eval, y_eval, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for label, clf_model in models_dict.items():
        if hasattr(clf_model, "predict_proba"):
            scores = clf_model.predict_proba(X_eval)[:, 1]
        elif hasattr(clf_model, "decision_function"):
            scores = clf_model.decision_function(X_eval)
        else:
            continue

        precision, recall, _ = precision_recall_curve(y_eval, scores)
        avg_precision = average_precision_score(y_eval, scores)
        plt.plot(recall, precision, label=f"{label} (AP={avg_precision:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"))
    plt.close()

# ==== Call PR curve function ====
generate_precision_recall_plot(models_to_train, X_test, y_test)
