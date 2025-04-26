import os
import shutil
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm

GERMAN_TO_ENGLISH_EMOTION_MAP = {
    'W': 'anger',         # Wut (Anger)
    'L': 'boredom',       # Langeweile (Boredom)
    'E': 'disgust',       # Ekel (Disgust)
    'A': 'fear',          # Angst (Fear)
    'F': 'happiness',     # Freude (Happiness)
    'T': 'sadness',       # Trauer (Sadness)
    'N': 'neutral'        # Neutral
}

SPEAKER_INFO = {
    "03": {"gender": "male", "age": 31},
    "08": {"gender": "female", "age": 34},
    "09": {"gender": "female", "age": 21},
    "10": {"gender": "male", "age": 32},
    "11": {"gender": "male", "age": 26},
    "12": {"gender": "male", "age": 30},
    "13": {"gender": "female", "age": 32},
    "14": {"gender": "female", "age": 35},
    "15": {"gender": "male", "age": 25},
    "16": {"gender": "female", "age": 31}
}

def extract_metadata(filename):
    """
    Extracts speaker ID and emotion label from filename.

    Args:
        filename (str): Name of the audio file.

    Returns:
        tuple: (speaker_id, emotion, gender, age) or None if the emotion is unknown.
    """
    try:
        speaker_id = filename[:2]
        german_emotion_code = filename[5]
        emotion = GERMAN_TO_ENGLISH_EMOTION_MAP.get(german_emotion_code)
        if emotion is None:
            print(f"⚠️ Warning: Skipping file with unknown emotion - {filename}")
            return None, None, None, None
        speaker_info = SPEAKER_INFO.get(speaker_id, {"gender": "unknown", "age": "unknown"})
        gender, age = speaker_info["gender"], speaker_info["age"]

        return speaker_id, emotion, gender, age

    except Exception as e:
        print(f"Error extracting metadata from {filename}: {e}")
        return None, None, None, None

def organize_dataset(dataset_path, output_dir="data/"):
    """
    Organizes the EMODB dataset into structured directories: `data/{emotion}/{speaker}/`.

    Args:
        dataset_path (str): Path to the dataset containing `.wav` files.
        output_dir (str): Path to save the organized dataset.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metadata = []

    print(f"Organizing dataset from {dataset_path} to {output_dir}...\n")

    for file in tqdm(os.listdir(dataset_path), desc="Processing Files"):
        if file.endswith(".wav"):
            speaker_id, emotion, gender, age = extract_metadata(file)
            if emotion is None or speaker_id is None:
                continue

            source_path = os.path.join(dataset_path, file)
            target_dir = os.path.join(output_dir, emotion, speaker_id)
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            shutil.move(source_path, os.path.join(target_dir, file))
            metadata.append({
                "file": file,
                "speaker": speaker_id,
                "emotion": emotion,
                "gender": gender,
                "age": age
            })
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print("Dataset successfully organized!\n")

def downsample_audio(input_dir, output_dir="data_16k/"):
    """
    Downsamples all `.wav` files to 16kHz for consistency.

    Args:
        input_dir (str): Path to the organized dataset.
        output_dir (str): Path to save the downsampled dataset.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downsampling audio files from {input_dir} to {output_dir}...\n")

    for subdir, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {subdir}"):
            if file.endswith(".wav"):
                source_path = os.path.join(subdir, file)
                target_path = source_path.replace(input_dir, output_dir)
                Path(os.path.dirname(target_path)).mkdir(parents=True, exist_ok=True)

                try:
                    y, sr = librosa.load(source_path, sr=16000)
                    sf.write(target_path, y, 16000)
                except Exception as e:
                    print(f" Error processing {file}: {e}")

    print(" Downsampling complete!\n")
if __name__ == "__main__":
    dataset_path = "/home/shirish/Phd/Coursework/sp4ml/archive/wav"

    organize_dataset(dataset_path)

    downsample_audio("data/")
