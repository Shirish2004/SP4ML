import torch
import numpy as np
import pandas as pd
import os
import yaml
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# ============================
# LOAD CONFIGURATION
# ============================
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config.yaml"))

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

class AudioDataset(Dataset):
    """
    Custom PyTorch Dataset for loading precomputed MFCCs / Mel Spectrograms / Fractional features.
    Uses `metadata.csv` and splits dataset into train/val/test dynamically.
    """

    def __init__(self, metadata_path, feature_dir, feature_type="both", split="train", transform=None):
        self.feature_dir = feature_dir
        self.feature_type = feature_type
        self.transform = transform
        self.max_length = config["features"]["max_length"]

        self.metadata = pd.read_csv(metadata_path)
        self.emotions = sorted(self.metadata["emotion"].unique())
        self.label_map = {emotion: idx for idx, emotion in enumerate(self.emotions)}

        data = []
        for _, row in self.metadata.iterrows():
            filename = row["file"].replace(".wav", "")
            emotion = row["emotion"]
            speaker = str(row["speaker"]).zfill(2)
            label = self.label_map[emotion]

            mfcc_path = os.path.join(feature_dir, emotion, speaker, f"{filename}_mfcc_ord1.0.npy")
            mel_path = os.path.join(feature_dir, emotion, speaker, f"{filename}_mel_ord1.0.npy")

            # Fractional
            frac_paths = []
            for alpha in config["experiment"]["fractional_orders"]:
                alpha_str = f"{alpha}".replace('.', '')
                fmfcc = os.path.join(feature_dir, emotion, speaker, f"{filename}_mfcc_ord{alpha}.npy")
                fmel = os.path.join(feature_dir, emotion, speaker, f"{filename}_mel_ord{alpha}.npy")
                if os.path.exists(fmfcc) and os.path.exists(fmel):
                    frac_paths.append((fmfcc, fmel, label))

            if feature_type == "mfcc" and os.path.exists(mfcc_path):
                data.append((mfcc_path, label))
            elif feature_type == "mel" and os.path.exists(mel_path):
                data.append((mel_path, label))
            elif feature_type == "both" and os.path.exists(mfcc_path) and os.path.exists(mel_path):
                data.append((mfcc_path, mel_path, label))
            elif feature_type == "fractional_mfcc":
                for alpha in config["experiment"]["fractional_orders"]:
                    fmfcc = os.path.join(feature_dir, emotion, speaker, f"{filename}_mfcc_ord{alpha}.npy")
                    if os.path.exists(fmfcc):
                        data.append((fmfcc, label))
            elif feature_type == "fractional_mel":
                for alpha in config["experiment"]["fractional_orders"]:
                    fmel = os.path.join(feature_dir, emotion, speaker, f"{filename}_mel_ord{alpha}.npy")
                    if os.path.exists(fmel):
                        data.append((fmel, label))
            elif feature_type == "combined_fractional":
                data.extend(frac_paths)

        if len(data) == 0:
            raise ValueError(f"No data found for feature_type='{feature_type}' in {feature_dir}")

        train_ratio = config["data"]["train_ratio"]
        val_ratio = config["data"]["val_ratio"]
        test_ratio = config["data"]["test_ratio"]

        labels = [x[1] if feature_type in ["mfcc", "mel", "fractional_mfcc", "fractional_mel"] else x[2] for x in data]

        try:
            train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), stratify=labels, random_state=config["experiment"]["seed"])
            val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (test_ratio + val_ratio)), stratify=[x[1] if feature_type in ["mfcc", "mel", "fractional_mfcc", "fractional_mel"] else x[2] for x in temp_data], random_state=config["experiment"]["seed"])
        except:
            train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), shuffle=True, random_state=config["experiment"]["seed"])
            val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (test_ratio + val_ratio)), shuffle=True, random_state=config["experiment"]["seed"])

        if split == "train":
            self.samples = train_data
        elif split == "val":
            self.samples = val_data
        else:
            self.samples = test_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.feature_type in ["mfcc", "mel", "fractional_mfcc", "fractional_mel"]:
            feature = np.load(sample[0])
            label = sample[1]
        else:
            mfcc = np.load(sample[0])
            mel = np.load(sample[1])
            label = sample[2]
            feature = np.concatenate((mfcc, mel), axis=0)

        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        feature = self.pad_tensor(feature)

        if self.transform:
            feature = self.transform(feature)

        return feature.unsqueeze(0), label

    def pad_tensor(self, tensor):
        num_features, time_steps = tensor.shape
        pad_size = self.max_length - time_steps
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size), "constant", 0)
        elif pad_size < 0:
            tensor = tensor[:, :self.max_length]
        return tensor
