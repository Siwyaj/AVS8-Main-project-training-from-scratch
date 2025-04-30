"""
This script will connect to the microphone and record s seconds of audio, save it to feed to the model.
While the model is working record the next s seconds of audio and save it to feed to the model.
This will be done in a loop until the user stops the program.
"""
import pyaudio #pip install pyaudio #https://realpython.com/playing-and-recording-sound-python/#pyaudio
import wave
import sounddevice as sd #python -m pip install sounddevice
import numpy as np
import os
import wavio #for saving the audio file

fs = 44100  # Sample rate
recording_session_name = "Test recordings"  # Name of the recording session
seconds = 1  # Number of seconds to record at a time

def RecordAudioSnippet(path, microphone_channel=2,seconds=1):
    """
    This function will record s seconds of audio and save it to a file.
    :param recordingNumber (int): The recording number to save the file as
    :param microphone_channels: The channels used for the recording
    :param seconds (int): The number of seconds to record at a time
    :return: wave file object
    """
    # Create a new file to save the audio snippet
    #filename = f"audio_snippet_{seconds}_seconds_{recordingNumber}.wav"
    #path = os.path.join("Recording", filename)
    print(f"Recording {seconds} seconds of audio...")

    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=microphone_channel)
    sd.wait()  # Wait until recording is finished
    print(f"Finished recording {seconds} seconds of audio.")
    
    # Save the audio snippet to a file
    wavio.write(path, audio, fs, sampwidth=2)  # Save as WAV file
    
    print(f"Saved audio snippet to {path}.")
    return audio

if __name__ == "__main__":
    # Record audio snippets in a loop
    recordinNumber = 0
    while True:
        recording_session_name = f"test_recordings_snippet_{seconds}_seconds_{recordinNumber}.wav"
        path = os.path.join("Recording", recording_session_name)
        try:
            recordinNumber += 1
            audio = RecordAudioSnippet(path, seconds=seconds)
        except KeyboardInterrupt:
            print("Recording stopped by user.")
            break