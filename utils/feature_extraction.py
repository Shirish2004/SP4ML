import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import yaml
from scipy.special import gamma

# Load config.yaml
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config.yaml"))
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FRACTIONAL_ORDERS = config["experiment"]["fractional_orders"]
SAVE_PLOTS = config["features"]["fractional"]["save_plots"]
SAMPLE_RATE = config["features"]["sample_rate"]

def create_directory(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def fractional_derivative(signal, alpha, max_terms=100):
    N = len(signal)
    frac_diff = np.zeros(N)
    coeffs = np.array([(-1)**k * gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1))
                       for k in range(max_terms)])
    for n in range(N):
        sum_val = 0.0
        for k in range(min(n + 1, max_terms)):
            sum_val += coeffs[k] * signal[n - k]
        frac_diff[n] = sum_val
    # return np.nan_to_num(frac_diff, nan=0.0, posinf=0.0, neginf=0.0)
    return frac_diff
    
def extract_mfcc(audio_path, save_path, alpha):
    if not os.path.isfile(audio_path):
        print(f"Skipping invalid file: {audio_path}")
        return
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    if alpha != 1.0:
        y = fractional_derivative(y, alpha)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config["features"]["mfcc"]["num_mfcc"])
    npy_path = os.path.join(save_path, f"{Path(audio_path).stem}_mfcc_ord{alpha}.npy")
    np.save(npy_path, mfcc)
    if SAVE_PLOTS:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time', cmap='viridis')
        plt.colorbar(label="MFCC Coefficients")
        plt.title(f"MFCC (alpha={alpha}) - {os.path.basename(audio_path)}")
        plt.xlabel("Time")
        plt.ylabel("MFCC Coefficients")
        plt.tight_layout()
        plt.savefig(npy_path.replace(".npy", ".png"))
        plt.close()

def extract_mel(audio_path, save_path, alpha):
    if not os.path.isfile(audio_path):
        print(f"Skipping invalid file: {audio_path}")
        return
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    if alpha != 1.0:
        y = fractional_derivative(y, alpha)
    mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                         n_mels=config["features"]["mel"]["n_mels"],
                                         fmin=config["features"]["mel"]["fmin"],
                                         fmax=config["features"]["mel"]["fmax"],
                                         power=config["features"]["mel"]["power"])
    mel_db = librosa.power_to_db(mel, ref=np.max)
    npy_path = os.path.join(save_path, f"{Path(audio_path).stem}_mel_ord{alpha}.npy")
    np.save(npy_path, mel_db)
    if SAVE_PLOTS:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', cmap='inferno')
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Mel Spectrogram (alpha={alpha}) - {os.path.basename(audio_path)}")
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency")
        plt.tight_layout()
        plt.savefig(npy_path.replace(".npy", ".png"))
        plt.close()

def process_dataset(dataset_path, save_dir, feature_type="both"):
    if not os.path.isdir(dataset_path):
        print(f"Invalid dataset path: {dataset_path}")
        return
    for emotion_dir in tqdm(os.listdir(dataset_path), desc="Processing Emotions"):
        emotion_path = os.path.join(dataset_path, emotion_dir)
        if not os.path.isdir(emotion_path):
            continue
        for speaker_dir in os.listdir(emotion_path):
            speaker_path = os.path.join(emotion_path, speaker_dir)
            if not os.path.isdir(speaker_path):
                continue
            audio_files = list(Path(speaker_path).rglob("*.wav"))
            if not audio_files:
                print(f"No .wav files found for {emotion_dir}/{speaker_dir}")
                continue
            save_path = os.path.join(save_dir, emotion_dir, speaker_dir)
            create_directory(save_path)
            for audio_path in tqdm(audio_files, desc=f"{emotion_dir}/{speaker_dir}"):
                audio_path = str(audio_path)
                if feature_type == "mfcc":
                    extract_mfcc(audio_path, save_path, alpha=1.0)
                elif feature_type == "mel":
                    extract_mel(audio_path, save_path, alpha=1.0)
                elif feature_type == "both":
                    extract_mfcc(audio_path, save_path, alpha=1.0)
                    extract_mel(audio_path, save_path, alpha=1.0)
                elif feature_type == "fractional_mfcc":
                    for alpha in FRACTIONAL_ORDERS:
                        extract_mfcc(audio_path, save_path, alpha)
                elif feature_type == "fractional_mel":
                    for alpha in FRACTIONAL_ORDERS:
                        extract_mel(audio_path, save_path, alpha)
                elif feature_type == "combined_fractional":
                    for alpha in FRACTIONAL_ORDERS:
                        extract_mfcc(audio_path, save_path, alpha)
                        extract_mel(audio_path, save_path, alpha)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract MFCCs, Mel Spectrograms, and Fractional Features")
    parser.add_argument("--input", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save extracted features.")
    parser.add_argument("--type", type=str, choices=["mfcc", "mel", "both", "fractional_mfcc", "fractional_mel", "combined_fractional"], default="both", help="Type of feature extraction")
    args = parser.parse_args()
    if os.path.isdir(args.input):
        process_dataset(args.input, args.output, feature_type=args.type)
    else:
        print(f"Error: {args.input} is not a valid directory.")
