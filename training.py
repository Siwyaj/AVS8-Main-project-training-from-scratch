"""
This script will contain the training of the model
This includes loading the data the right way, importing the model, validating the model(by validation and testing), and saving the model. for later use

For now it will seem that the model is trained by looking at the audio data together with the midi data.
The audio data is loaded into a spectrogram and use a CNN to predict the midi data.

Further understanding in spectrogram and midi data is needed to understand the model


"""

import numpy as np

data_path = "maestro-v3.0.0"

def load_data_HDF5(path):
    """
    This function will load the data from the path
    It will return the data as the HDF5 format
    """
    pass


