import os
import yaml
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from utils.dataset import AudioDataset

# Load config
CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

feature_type = config["experiment"]["feature_type"]
fractional_orders = config["experiment"].get("fractional_orders", [1.0])
feature_dir = config["data"]["features_dir"]
csv_log_path = config["logging"]["csv_log_path"]
os.makedirs(csv_log_path, exist_ok=True)

device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

def evaluate_model(model, dataloader, model_name, feature_name, split_name="Validation"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in dataloader:
            if features.shape[1] == 1:
                features = features.repeat(1, 3, 1, 1)
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f" {split_name} | {model_name} | {feature_name} → Acc: {acc:.4f}, F1: {f1:.4f}")
    return acc, prec, rec, f1, all_labels, all_preds

def plot_confusion_matrix(labels, preds, model_name, feature_name, split_name):
    cm = confusion_matrix(labels, preds)
    labels_unique = sorted(set(labels))
    df_cm = pd.DataFrame(cm, index=labels_unique, columns=labels_unique)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - {split_name} Confusion Matrix")

    save_dir = f"logs/confusion_matrices/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_{feature_name}_{split_name}_cm.png")
    plt.savefig(save_path)
    plt.close()

def main(test_mode=False):
    split = "test" if test_mode else "val"
    csv_metric_path = os.path.join(csv_log_path, f"eval_metrics_{split}.csv")
    report_dir = "logs/reports/"
    os.makedirs(report_dir, exist_ok=True)

    with open(csv_metric_path, "w") as log_file:
        log_file.write("Model,Feature,Accuracy,Precision,Recall,F1\n")

    for model_name in config["model"]["architectures"]:
        order_list = fractional_orders if "fractional" in feature_type else [1.0]

        for alpha in order_list:
            feature_name = f"{feature_type}_ord{alpha}" if alpha != 1.0 else feature_type

            print(f"\n🔍 Evaluating {model_name} on {feature_name} ({split})")

            dataset = AudioDataset(
                metadata_path=config["data"]["metadata_path"],
                feature_dir=feature_dir,
                feature_type=feature_type,
                split=split
            )
            loader = DataLoader(dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False)

            model_fn = getattr(models, model_name, None)
            if not model_fn:
                print(f" Model {model_name} not found.")
                continue

            model = model_fn(pretrained=False)
            if "resnet" in model_name:
                model.fc = nn.Linear(model.fc.in_features, config["experiment"]["num_classes"])
            elif "densenet" in model_name or "efficientnet" in model_name:
                model.classifier = nn.Linear(model.classifier.in_features, config["experiment"]["num_classes"])
            model = model.to(device)

            model_path = os.path.join(config["training"]["checkpoint_path"], f"{model_name}_{feature_name}_best.pth")
            if not os.path.exists(model_path):
                print(f" Model checkpoint not found at {model_path}")
                continue
            model.load_state_dict(torch.load(model_path))

            acc, prec, rec, f1, labels, preds = evaluate_model(model, loader, model_name, feature_name, split)

            # TensorBoard logging
            writer = SummaryWriter(log_dir=f"runs/eval/{model_name}_{feature_name}_{split}")
            writer.add_scalar("Accuracy", acc)
            writer.add_scalar("Precision", prec)
            writer.add_scalar("Recall", rec)
            writer.add_scalar("F1 Score", f1)
            writer.close()

            # CSV logging
            with open(csv_metric_path, "a") as log_file:
                log_file.write(f"{model_name},{feature_name},{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f}\n")

            # Confusion Matrix
            if config["evaluation"]["confusion_matrix"]:
                plot_confusion_matrix(labels, preds, model_name, feature_name, split)

            # Classification Report CSV
            if config["evaluation"]["save_csv_report"]:
                report = classification_report(labels, preds, output_dict=True)
                df = pd.DataFrame(report).transpose()
                df.to_csv(os.path.join(report_dir, f"{model_name}_{feature_name}_{split}_report.csv"))

if __name__ == "__main__":
    main(test_mode=False)
