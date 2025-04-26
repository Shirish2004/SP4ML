import argparse
import yaml
import os
from utils.feature_extraction import process_dataset
from models import train
import evaluate

# Load config
CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# CLI Arguments
parser = argparse.ArgumentParser(description="Main script for Audio Emotion Classification")

parser.add_argument(
    "--mode",
    type=str,
    choices=["extract", "train", "evaluate", "test"],
    required=True,
    help="Operation mode: extract, train, evaluate, or test"
)

parser.add_argument(
    "--feature_type",
    type=str,
    choices=["mfcc", "mel", "both", "fractional_mfcc", "fractional_mel", "combined_fractional"],
    default=config["experiment"]["feature_type"],
    help="Feature type for extraction and training"
)

args = parser.parse_args()

# Update config based on CLI
config["experiment"]["feature_type"] = args.feature_type

# =============== FEATURE EXTRACTION ===============
if args.mode == "extract":
    print(f"\n🔹 Extracting features of type: {args.feature_type}")
    process_dataset(
        dataset_path=config["data"]["dataset_root"],
        save_dir=config["data"]["features_dir"],
        feature_type=args.feature_type
    )
    print(" Feature extraction complete.\n")

# =============== TRAINING =========================
elif args.mode == "train":
    print(f"\n Starting training with feature type: {args.feature_type}")
    # train.main()
    print(" Training complete.\n")

# =============== EVALUATION ========================
elif args.mode == "evaluate":
    print(f"\n Evaluating model(s) with feature type: {args.feature_type}")
    evaluate.main(test_mode=False)
    print(" Evaluation complete.\n")

# =============== TESTING ==========================
elif args.mode == "test":
    print(f"\nTesting model(s) with feature type: {args.feature_type}")
    evaluate.main(test_mode=True)
    print("Testing complete.\n")

# =============== INVALID ==========================
else:
    print(" Invalid mode selected. Choose from: extract / train / evaluate / test.")
