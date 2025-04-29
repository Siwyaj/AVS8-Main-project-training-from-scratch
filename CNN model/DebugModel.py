import torch
import torchlibrosa
import librosa
import matplotlib.pyplot as plt
import numpy as np

# Function to generate and save the spectrogram images
def save_spectrogram_image(spectrogram, filename, title, xlabel="Time [s]", ylabel="Frequency [Hz]"):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Spectrogram visualization function
def visualize_spectrogram(wav_path='testWav.wav', sample_rate=16000, window_size=2048, hop_size=512, mel_bins=229):
    # Load audio file using librosa
    y, sr = librosa.load(wav_path, sr=sample_rate)
    sample_rate = sr  # Update sample rate to the one used in librosa
    
    # Convert the audio to a tensor and add a batch dimension
    y_tensor = torch.tensor(y).unsqueeze(0)  # Shape: (1, N) where N is the number of samples (time axis)

    # ------------------- Log-Mel Spectrogram using torchlibrosa -------------------
    logmel_extractor = torchlibrosa.stft.LogmelFilterBank(  # Removed `sample_rate` argument
        n_fft=window_size,
        n_mels=mel_bins,
        fmin=30,
        fmax=sample_rate // 2,
        amin=1e-10,
        top_db=None
    )

    # Apply the spectrogram extractor
    spectrogram = torchlibrosa.stft.Spectrogram(n_fft=window_size, hop_length=hop_size)(y_tensor)
    
    # Apply the log-mel extractor
    logmel = logmel_extractor(spectrogram)
    
    # Remove the extra dimensions (channels) for visualization
    logmel = logmel.squeeze().cpu().numpy()  # Shape: (mel_bins, time)
    save_spectrogram_image(logmel, 'logmel_spectrogram.png', 'Log-Mel Spectrogram')

    # ------------------- Power Spectrogram using librosa -------------------
    # Generate the power spectrogram with librosa
    D = librosa.stft(y, n_fft=window_size, hop_length=hop_size, window='hann')
    power_spec = np.abs(D)**2  # Power spectrogram

    # Convert to decibels (dB) for better visualization
    power_spec_db = librosa.amplitude_to_db(power_spec, ref=np.max)
    
    # Time axis for plotting
    times = np.arange(power_spec_db.shape[1]) * hop_size / sample_rate

    save_spectrogram_image(power_spec_db, 'power_spectrogram.png', 'Power Spectrogram', xlabel="Time [s]", ylabel="Frequency [Hz]")

    # ------------------- Mel Spectrogram using librosa -------------------
    # Generate mel spectrogram using librosa
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=window_size, hop_length=hop_size, n_mels=mel_bins, fmin=30, fmax=sample_rate // 2)

    # Convert to decibels (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot the Mel spectrogram
    save_spectrogram_image(mel_spec_db, 'mel_spectrogram.png', 'Mel Spectrogram')

# Call the function to visualize all spectrograms
visualize_spectrogram('testWav.wav')
