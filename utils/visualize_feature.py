import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

def plot_signal(signal, sr, title="Signal Waveform"):
    """Plot the raw waveform of an audio signal."""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def plot_fft(signal, sr, title="FFT Spectrum"):
    """Compute and plot the FFT spectrum."""
    fft_spectrum = np.abs(np.fft.rfft(signal))  # Compute FFT magnitude
    freqs = np.fft.rfftfreq(len(signal), d=1/sr)  # Compute frequency bins

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_spectrum, color='red')
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, sr/2)
    plt.show()

def plot_spectrogram(signal, sr, title="Spectrogram (STFT)"):
    """Compute and plot the Short-Time Fourier Transform (STFT)."""
    stft = np.abs(librosa.stft(signal))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def plot_mel_filter_bank(sr, n_fft=2048, n_mels=40):
    """Plot the Mel filter bank."""
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    plt.figure(figsize=(10, 4))
    for i in range(n_mels):
        plt.plot(mel_filters[i], label=f"Filter {i+1}")
    plt.title("Mel Filter Bank")
    plt.xlabel("FFT Bins")
    plt.ylabel("Amplitude")
    plt.show()

def plot_mel_spectrogram(signal, sr, title="Mel Spectrogram"):
    """Compute and plot the Mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()
    
def plot_mfcc(signal, sr, title="MFCC"):
    """Compute and plot MFCCs step by step."""
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Step 1: Apply DCT to decorrelate
    mfccs = scipy.fftpack.dct(mel_spec_db, axis=0, norm='ortho')
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
    plt.colorbar(label="MFCC Coefficients")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.show()

def process_audio(audio_path):
    """Main function to compute all steps for MFCC & Mel Spectrogram."""
    signal, sr = librosa.load(audio_path, sr=None)
    plot_signal(signal, sr, "Raw Audio Waveform")
    plot_fft(signal, sr, "FFT Spectrum of Signal")
    plot_spectrogram(signal, sr, "Spectrogram (STFT)")
    plot_mel_filter_bank(sr)
    plot_mel_spectrogram(signal, sr, "Mel Spectrogram")
    plot_mfcc(signal, sr, "MFCC Coefficients")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize MFCC & Mel Spectrogram Step by Step")
    parser.add_argument("--input", type=str, required=True, help="Path to the audio file.")
    
    args = parser.parse_args()
    process_audio(args.input)
