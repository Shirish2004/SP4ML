import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import csv
from utils.dataset import AudioDataset

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

feature_type = config["experiment"]["feature_type"]
feature_dir = config["data"]["features_dir"]
fractional_orders = config["experiment"].get("fractional_orders", [1.0])
log_dir = config["logging"]["tensorboard_logdir"]
csv_log_path = config["logging"]["csv_log_path"]
os.makedirs(log_dir, exist_ok=True)
os.makedirs(csv_log_path, exist_ok=True)
os.makedirs(config['training']['checkpoint_path'], exist_ok=True)

device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# Training loop for each model + each fractional order
for model_name in config["model"]["architectures"]:
    order_list = fractional_orders if "fractional" in feature_type else [1.0]

    for alpha in order_list:
        ft_suffix = f"{feature_type}_ord{alpha}" if alpha != 1.0 else feature_type
        print(f"\n🔹 Training {model_name} on {ft_suffix}...\n")

        # TensorBoard setup
        writer = SummaryWriter(log_dir=os.path.join(log_dir, f"{model_name}_{ft_suffix}"))

        # Load Dataset
        train_dataset = AudioDataset(
            metadata_path=config["data"]["metadata_path"],
            feature_dir=feature_dir,
            feature_type=feature_type,
            split="train"
        )
        val_dataset = AudioDataset(
            metadata_path=config["data"]["metadata_path"],
            feature_dir=feature_dir,
            feature_type=feature_type,
            split="val"
        )

        train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

        # Model setup
        model_fn = getattr(models, model_name)
        model = model_fn(weights="DEFAULT") if config["model"]["pretrained"] else model_fn()

        if "resnet" in model_name:
            model.fc = nn.Linear(model.fc.in_features, config["experiment"]["num_classes"])
        elif "densenet" in model_name or "efficientnet" in model_name:
            model.classifier = nn.Linear(model.classifier.in_features, config["experiment"]["num_classes"])

        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]),
                               weight_decay=float(config["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

        best_val_acc = 0
        early_stop_counter = 0

        # CSV Log File Setup
        csv_path = os.path.join(csv_log_path, f"{model_name}_{ft_suffix}_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "LR"])

        # Training
        for epoch in range(config["training"]["num_epochs"]):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for features, labels in train_loader:
                if features.shape[1] == 1:
                    features = features.repeat(1, 3, 1, 1)
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += labels.size(0)

            train_acc = 100 * train_correct / train_total

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for features, labels in val_loader:
                    if features.shape[1] == 1:
                        features = features.repeat(1, 3, 1, 1)
                    features, labels = features.to(device), labels.to(device)

                    outputs = model(features)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = 100 * val_correct / val_total
            scheduler.step(val_acc)

            # Log TensorBoard
            writer.add_scalars(f"{model_name}/{ft_suffix}", {
                "Train Loss": train_loss / len(train_loader),
                "Train Accuracy": train_acc,
                "Val Loss": val_loss / len(val_loader),
                "Val Accuracy": val_acc
            }, epoch + 1)

            # Log CSV
            with open(csv_path, "a", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([
                    epoch + 1,
                    round(train_loss / len(train_loader), 4),
                    round(train_acc, 2),
                    round(val_loss / len(val_loader), 4),
                    round(val_acc, 2),
                    optimizer.param_groups[0]['lr']
                ])

            print(f"Epoch {epoch + 1:02d}: "
                  f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(config["training"]["checkpoint_path"],
                                          f"{model_name}_{ft_suffix}_best.pth")
                torch.save(model.state_dict(), model_path)
                early_stop_counter = 0
                print(f" Best model saved: {model_path}")
            else:
                early_stop_counter += 1

            # Early Stopping
            if config["training"]["early_stopping"]["enabled"] and \
                    early_stop_counter >= config["training"]["early_stopping"]["patience"]:
                print("Early stopping triggered.")
                break

        writer.close()
        print(f" Training complete for {model_name} on {ft_suffix}")

