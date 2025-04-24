"""
This script contains the function which will convert a .wav to spectrogram.
If run as main it will convert a .wav file to a spectrogram and save it as a .png file.
"""


def PathWavToSpectrogram(wavFilePath):
    """
    Converts a .wav file to a spectrogram and saves it as a .png file.
    :param wavFile: The path to the .wav file.
    :return: None
    """
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    import numpy as np
    import librosa
    import librosa.display
    import os

    # Load the .wav file
    y, sr = librosa.load(wavFilePath, sr=None)
    print(f"Loaded {wavFilePath} with sample rate {sr}")

    # Create a mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert to log scale (dB)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot and save the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

    # Save the spectrogram as a .png file
    output_path = os.path.splitext(wavFilePath)[0] + '_spectrogram.png'
    plt.savefig(output_path)
    plt.close()



if __name__ == '__main__':
    PathWavToSpectrogram('testWav.wav')