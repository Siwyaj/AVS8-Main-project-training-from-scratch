"""
This script will take the raw data from maestro and convert it to HDF5 file
"""

import os
#os.add_dll_directory(r"C:/AVS8 Main project/tools/fluidsynth/bin")

import numpy as np
import h5py # for HDF5 file handling
import librosa # for audio processing
import mido # for MIDI file processing
import DataLoadingConfig as config
import note_seq
import pretty_midi


#data paths
hd5f_path = "maestro-v3.0.0-hdf5.hdf5"
data_path = "maestro-v3.0.0"
csv_path = os.path.join(data_path, "maestro-v3.0.0.csv")

#from csv, read structure
#use midi_filename as number of samples
#read csv file
with open(csv_path, 'r', encoding='utf-8') as csv_file:
    csv_lines = csv_file.readlines()


def float32_to_int16(x):
  return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
  return (x / 32767.).astype(np.float32)

sample_rate = 16000

with h5py.File(os.path.join(data_path, f'{year}.h5'), "w") as f:
  audio, _ = librosa.core.load(os.path.join(data_path, audio_path), sr=sample_rate, mono=True)
  ns = note_seq.midi_file_to_note_sequence(os.path.join(data_path, midi_path))
  group = f.create_group(filename)
  group.create_dataset('audio', data=float32_to_int16(audio))
  group.create_dataset('midi', data=np.void(ns.SerializeToString()))

with h5py.File(hd5f_path, 'w') as f:


    for n in range(1,len(csv_lines)):
        csv_lines[n] = csv_lines[n].strip().split(',')
        #0 = canonical_composer
        #1 = canonical_title
        #2 = split
        #3 = year
        #4 = midi_filename
        #5 = audio_filename
        #6 = duration

        group = f.create_group(csv_lines[n][4])
        audio, _ = librosa.core.load(os.path.join(data_path, csv_lines[n][5]), sr=sample_rate, mono=True)
        ns = note_seq.midi_file_to_note_sequence(os.path.join(data_path, csv_lines[n][4]))
        group.create_dataset('audio', data=float32_to_int16(audio))
        group.create_dataset('midi', data=np.void(ns.SerializeToString()))


        break
csv_file.close()
