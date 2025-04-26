import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.special import gamma

def fractional_derivative(signal, alpha=0.9, max_terms=100):
    """Apply Grünwald–Letnikov fractional derivative."""
    N = len(signal)
    frac_diff = np.zeros(N)
    coeffs = np.array([(-1)**k * gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1)) for k in range(max_terms)])
    for n in range(N):
        sum_val = 0.0
        for k in range(min(n + 1, max_terms)):
            sum_val += coeffs[k] * signal[n - k]
        frac_diff[n] = sum_val
    return np.nan_to_num(frac_diff, nan=0.0, posinf=0.0, neginf=0.0)

def plot_signal(signal, sr, title="Signal Waveform"):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_fft(signal, sr, title="FFT Spectrum"):
    fft_spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1/sr)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_spectrum, color='red')
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, sr/2)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(signal, sr, title="Spectrogram (STFT)"):
    stft = np.abs(librosa.stft(signal))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_mel_filter_bank(sr, n_fft=2048, n_mels=40):
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    plt.figure(figsize=(10, 4))
    for i in range(n_mels):
        plt.plot(mel_filters[i])
    plt.title("Mel Filter Bank")
    plt.xlabel("FFT Bins")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_mel_spectrogram(signal, sr, title="Fractional Mel Spectrogram"):
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_mfcc(signal, sr, title="Fractional MFCC"):
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mfccs = scipy.fftpack.dct(mel_spec_db, axis=0, norm='ortho')

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
    plt.colorbar(label="MFCC Coefficients")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show()

def process_fractional_audio(audio_path, alpha=0.9):
    signal, sr = librosa.load(audio_path, sr=None)
    y_frac = fractional_derivative(signal, alpha)

    plot_signal(y_frac, sr, f"Fractional Waveform (α={alpha})")
    plot_fft(y_frac, sr, f"Fractional FFT Spectrum (α={alpha})")
    plot_spectrogram(y_frac, sr, f"Fractional STFT Spectrogram (α={alpha})")
    plot_mel_filter_bank(sr)
    plot_mel_spectrogram(y_frac, sr, f"Fractional Mel Spectrogram (α={alpha})")
    plot_mfcc(y_frac, sr, f"Fractional MFCC Coefficients (α={alpha})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Fractional MFCC & Mel Spectrogram")
    parser.add_argument("--input", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--alpha", type=float, default=0.9, help="Fractional order (default: 0.9)")
    args = parser.parse_args()

    process_fractional_audio(args.input, alpha=args.alpha)
