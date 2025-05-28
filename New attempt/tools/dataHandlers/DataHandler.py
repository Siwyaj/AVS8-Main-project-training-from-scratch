import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import Config.Config as Config
from utilities.utilities import TargetProcessor, int16_to_float32


class MaestroHDF5Dataset(object):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.segment_seconds = Config.segment_seconds
        self.hop_seconds = Config.hop_seconds
        self.sample_rate = Config.sample_rate
        self.frames_per_second = Config.frames_per_second
        self.begin_note = Config.begin_note
        self.classes_num = Config.classes_num

        self.segment_samples = int(self.sample_rate * self.segment_seconds)
        #self.augmentor = augmentor

        self.target_processor = TargetProcessor(self.segment_seconds, 
            self.frames_per_second, self.begin_note, self.classes_num)


    def __getitem__(self, file, start_time):

        data_dict = {}
        hdf5_path = self.hdf5_path #get path of file
        with h5py.File(hdf5_path, 'r') as hf: #read file
            start_sample = int(start_time * self.sample_rate) #where does the segment start
            end_sample = start_sample + self.segment_samples #where does the segment end

            if end_sample >= hf['waveform'].shape[0]: #if the segment end is greater than the length of the audio file
                    start_sample -= self.segment_samples #start sample is set to the end sample minus the segment length
                    end_sample -= self.segment_samples #end sample is set to the start sample minus the segment length

            waveform = int16_to_float32(hf['waveform'][start_sample : end_sample]) #waveform is extracted from the file

            
            #add augementation here


            data_dict['waveform'] = waveform #add waveform to data dict
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]

            # Process MIDI events to target
            (target_dict, note_events, pedal_events) = \
                self.target_processor.process(start_time, midi_events_time, 
                    midi_events, extend_pedal=True, note_shift=0)
            
            for key in target_dict.keys():
                data_dict[key] = target_dict[key]

        return data_dict

class Sampler(object):
    '''
        The sampler class is used to sample the data from the HDF5 file.

        __init_:
        It takes the split (train, val, test) and gets the data from the HDF5 file with set split.
        The class then creates a list of segments [audio path, second_begin] and stores it in self.segment_list.
        the index of the list is then shuffled.

        __iter__:
        handels the batch logic used by dataLoader
        '''
    def __init__(self, hdf5_path, split='train'):
        
        self.hdf5_path = hdf5_path
        self.split = split
        self.segment_seconds = Config.segment_seconds
        self.hop_seconds = Config.hop_seconds
        self.sample_rate = Config.sample_rate
        self.frames_per_second = Config.frames_per_second
        self.batch_size = Config.batch_size

        


        self.segment_list = []
        with h5py.File(hdf5_path, 'r') as f:
            for key in f.keys():
                group = f[key]  # Access the actual group or dataset, not just the string key
                for attr in group.keys():
                    print(attr)
                if group.attrs['split'].decode() == split:
                    audio_name = group.attrs['audio_path'].decode()
                    start_time = 0
                    while start_time < group.attrs['duration']:  # Assuming duration is in root attributes
                        self.segment_list.append([audio_name, start_time])
                        start_time += self.hop_seconds
        
        self.pointer = 0
        self.segment_index = np.arange(len(self.segment_list))
        np.random.shuffle(self.segment_index)

    def __iter__(self):
        while True:
            batch_segment_list = []
            i=0
            while i < self.batch_size:
                index = self.segment_index[self.pointer]
                self.pointer += 1
                if self.pointer >= len(self.segment_index):
                    self.pointer = 0
                    np.random.shuffle(self.segment_index)
                batch_segment_list.append(self.segment_list[index])
                i+=1
            yield batch_segment_list
    
    def __len__(self):
        return -1
        
    def state_dict(self):
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
            
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']



def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict