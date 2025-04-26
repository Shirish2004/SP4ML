Here’s a **complete, highly detailed README.md** you can use directly for your project:

---

# 📚 README: **Fractional-Order Feature Extraction and Emotion Classification**

---

## 🔷 Project Title

**Memory-Enhanced Audio Classification using Fractional Calculus-based Features**

---

## 🔷 Project Description

This project aims to improve speech-based emotion classification by leveraging **fractional-order calculus**. Unlike conventional MFCCs and Mel spectrograms that rely on integer-order signal derivatives, this approach uses **fractionally differentiated signals** (FrMFCC and FrMel), capturing long-term memory effects and temporal evolution in speech.

---

## 🔷 Folder Structure

```bash
sp4ml/
│
├── main.py              # Main controller script
├── config.yaml          # Global configuration file
│
├── models/
│   └── train.py         # Model training script
│
├── utils/
│   ├── dataset.py       # Custom dataset loader (supports integer and fractional features)
│   ├── feature_extraction.py # Feature extraction (MFCC, Mel, FrMFCC, FrMel)
│   ├── organize_data.py # Organizes and preprocesses EMO-DB dataset
│
├── logs/                # TensorBoard logs, confusion matrices, evaluation reports
├── outputs/             # Directory where trained model checkpoints are saved
└── data/                # Organized and downsampled audio dataset
    └── features/        # Extracted features (.npy)
```

---

## 🔷 Dataset

- **Berlin Emotional Speech Database (EMO-DB)**:
  - 535 utterances by 10 professional speakers (5 male, 5 female).
  - 7 emotional classes: anger, boredom, anxiety, happiness, sadness, disgust, neutral.
  - Original sampling rate: 48kHz, downsampled to 16kHz for experiments.
- Dataset organization is handled via `organize_data.py`.

---

## 🔷 Feature Extraction

### ✅ Supported Types
- **Integer-order**:
  - MFCC
  - Mel Spectrogram
- **Fractional-order** (Fractional Differentiation with α = 0.9):
  - FrMFCC (Fractional MFCC)
  - FrMel (Fractional Mel Spectrogram)

### ✅ How It Works
- Compute **fractional derivative** of input signal using Grünwald–Letnikov approximation.
- Apply **standard MFCC** or **Mel spectrogram pipeline** on the fractionally modified signal.
- Save features (`.npy`) and corresponding plots (`.png`) for visualization.

### ✅ Fractional orders supported:
- Configured via `config.yaml` under `experiment -> fractional_orders`.
- Example: `[0.9]` for current study.

---

## 🔷 Requirements

```bash
pip install -r requirements.txt
```

Example libraries:
- torch
- torchvision
- librosa
- matplotlib
- numpy
- pandas
- tensorboard
- tqdm
- scikit-learn

---

## 🔷 Configuration (config.yaml)

- `experiment.feature_type`: Choose between `mfcc`, `mel`, `both`, `fractional_mfcc`, `fractional_mel`, `combined_fractional`
- `features.sample_rate`: 16000 Hz
- `features.max_length`: 100 frames
- `models.architectures`: resnet18, resnet34, resnet50, densenet121, densenet161, etc.
- `logging.tensorboard`: True
- `training.early_stopping.patience`: 5

---

## 🔷 How to Run Experiments

### 1️⃣ Organize Dataset
```bash
python utils/organize_data.py
```

### 2️⃣ Extract Features

**Integer-order (normal MFCC+Mel)**:
```bash
python utils/feature_extraction.py --input data/ --output data/features --type both
```

**Fractional-order (FrMFCC+FrMel)**:
```bash
python utils/feature_extraction.py --input data/ --output data/features --type combined_fractional
```

### 3️⃣ Train Models
```bash
python main.py --mode train --feature_type both
```
or for fractional features:
```bash
python main.py --mode train --feature_type combined_fractional
```

Models will be saved under:
```bash
outputs/checkpoints/
```

TensorBoard logs will be saved under:
```bash
logs/tensorboard/
```

---

### 4️⃣ Evaluate Trained Models
```bash
python main.py --mode evaluate --feature_type both
```
or
```bash
python main.py --mode evaluate --feature_type combined_fractional
```

Evaluation metrics (Accuracy, Precision, Recall, F1-score) and confusion matrices will be saved under:
```bash
logs/
```

---

## 🔷 Important Highlights

| Feature Type         | Description                  |
|----------------------|-------------------------------|
| MFCC / Mel            | Traditional signal-based     |
| FrMFCC / FrMel (α=0.9) | Memory-enhanced signal-based  |

- Fractional calculus introduces **memory** to signal representation.
- Better captures emotional patterns that unfold over time.

---

## 🔷 Key Mathematical Concepts

- **Gamma Function**: Extends factorial to non-integer orders, crucial for fractional binomial coefficients.
- **Grünwald–Letnikov Fractional Derivative**:
  \[
  D^\alpha f(t) = \lim_{h \to 0} \frac{1}{h^\alpha} \sum_{k=0}^{\infty} (-1)^k \binom{\alpha}{k} f(t - kh)
  \]

- **FrMFCC Pipeline**:
  - Fractional Differentiation → STFT → Mel Filterbanks → Log Compression → DCT

- **FrMel Pipeline**:
  - Fractional Differentiation → STFT → Mel Filterbanks → Log Compression

---

## 🔷 Results Summary

| Model      | Integer Order Accuracy | Fractional Order Accuracy (α=0.9) |
|------------|-------------------------|----------------------------------|
| DenseNet121 | 87.17%                   | 79.49%                           |
| DenseNet161 | 83.33%                   | 78.20%                           |
| ResNet18    | 82.05%                   | 78.20%                           |
| ResNet34    | 78.20%                   | 80.77%                           |
| ResNet50    | 84.62%                   | 74.35%                           |

✅ In some models (e.g., ResNet34), **fractional features even outperform** normal features, suggesting model sensitivity to memory information.

---

## 🔷 Future Work Suggestions

- Tune fractional orders dynamically instead of fixed α.
- Explore more memory-aware architectures (like Temporal CNNs).
- Extend to multilingual emotional speech datasets.

---

## 🔷 Author

- **Shirish Shekhar Jha**
- PhD Student, IISER Bhopal
- Specializing in Robotics, Machine Learning,and their derivatives.

---
