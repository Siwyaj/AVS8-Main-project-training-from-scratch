'''
This file reads the csv file in the maestro dataset and creates a hdf5 file with the following structure:
    maestro-v3.0.0-hdf5.hdf5
    ├── year
    │   ├── midi_filename
    │   │   ├── midi_path
    │   │   ├── audio_path
    │   │   ├── composer
    │   │   ├── title
    │   │   ├── split
    │   │   ├── year
    │   │   ├── duration
'''

import os
import h5py
import numpy as np
import csv
import librosa

from tools.ReadMidiPath import ReadMidi
import Config.Config as Config

def MakeHDF5():
    #get path to the dataset

    maestro_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", 'dataset'))
    maestro_CSV_path = os.path.join(maestro_dataset_path, 'maestro-v3.0.0.csv')
    hdf5_path = os.path.join('maestro-v3.0.0-new_attempt-hdf5.hdf5')
    #delete the HDF5 file if it already exists
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)

    #convert dataset to HDF5

    #read CSV for line count and path
    #with open(maestro_CSV_path, 'r', encoding='utf-8') as csv_file:
    #    csv_lines = csv_file.readlines()

    with open(maestro_CSV_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)  # Automatically handles quoted values
        csv_lines = list(csv_reader)
        print(len(csv_lines))

        for row in csv_reader:
            print(row)  # Now quoted values stay intact

    csv_lines_count = len(csv_lines)

    with h5py.File(hdf5_path, 'w') as f:
        for file in range(1,csv_lines_count):
            csv_line_split = csv_lines[file]
            #print(csv_line_split)
            #0 = canonical_composer
            #1 = canonical_title
            #2 = split
            #3 = year
            #4 = midi_filename
            #5 = audio_filename
            #6 = duration
            year = csv_line_split[3]

            if year not in f:
                year_group = f.create_group(year)
            else:
                year_group = f[year]

            midi_path = os.path.join(maestro_dataset_path,"maestro-v3.0.0", csv_line_split[4])
            audio_path = os.path.join(maestro_dataset_path,"maestro-v3.0.0", csv_line_split[5])

            print("csv_midi_path:", midi_path   )
            only_midi_name = csv_line_split[4].split('/')[-1]
            print("currently working on:", only_midi_name, "counter:", file)
            midi_group = year_group.create_group(only_midi_name)



            #store as HDF5
            #group = f.create_group(csv_line_split[4])
            #print(f"Creating group: {csv_line_split[4]}")
            #print(group)
            midi_group.create_dataset('midi_path', data=midi_path)
            midi_group.create_dataset('audio_path', data=audio_path)
            midi_group.create_dataset('composer', data=csv_line_split[0], dtype=h5py.string_dtype())
            #print(f"Storing composer: {csv_line_split[0]}")
            midi_group.create_dataset('title', data=csv_line_split[1], dtype=h5py.string_dtype())
            midi_group.create_dataset('split', data=csv_line_split[2], dtype=h5py.string_dtype())
            midi_group.create_dataset('year', data=csv_line_split[3], dtype=h5py.string_dtype())
            midi_group.create_dataset('duration', data=csv_line_split[6], dtype=h5py.string_dtype())
            #print("HDF5 Groups:", list(f.keys()))

            #midi file as dictionary
            midi_dict = ReadMidi(midi_path)
            midi_group.create_dataset(name = 'midi_event', data = [e.encode() for e in midi_dict["midi_event"]], dtype='S100')
            midi_group.create_dataset(name = 'midi_event_time', data = midi_dict["midi_event_time"], dtype=np.float32)

            #wav file
            (audio, _) = librosa.core.load(audio_path, sr=Config.sample_rate, mono=True)
            midi_group.create_dataset(name="waveform", data=np.float16(audio), dtype=np.int16)

        csv_file.close()
        f.close()


if __name__ == '__main__':
    MakeHDF5()