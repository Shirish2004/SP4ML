import os
import matplotlib.pyplot as plt
import pandas as pd

log_dir = "logs/"  

log_files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]

results = {}

for log_file in log_files:
    model_name = log_file.replace(".csv", "") 
    df = pd.read_csv(os.path.join(log_dir, log_file))  
    results[model_name] = df  
color_map = {
    "resnet18": "blue",
    "resnet34": "cyan",
    "resnet50": "purple",
    "resnet101": "green",
    "resnet152": "lightblue",
    "densenet121": "red",
    "densenet161": "brown",
    "densenet169": "orange",
    "densenet201": "yellow",
}

# ===============================
# Train Loss Plot
# ===============================
plt.figure(figsize=(10, 5))
for model, df in results.items():
    plt.plot(df["Epoch"], df["Train Loss"], label=model, color=color_map.get(model, "black"))
plt.xlabel("Epochs")
plt.ylabel("Train Loss")
plt.title("Train Loss Over Time")
plt.legend()
plt.grid()
plt.show()

# ===============================
# Validation Loss Plot
# ===============================
plt.figure(figsize=(10, 5))
for model, df in results.items():
    plt.plot(df["Epoch"], df["Validation Loss"], label=model, color=color_map.get(model, "black"))
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Over Time")
plt.legend()
plt.grid()
plt.show()

# ===============================
# Accuracy Plot
# ===============================
plt.figure(figsize=(10, 5))
for model, df in results.items():
    plt.plot(df["Epoch"], df["Validation Accuracy"], label=model, color=color_map.get(model, "black"))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Time")
plt.legend()
plt.grid()
plt.show()
