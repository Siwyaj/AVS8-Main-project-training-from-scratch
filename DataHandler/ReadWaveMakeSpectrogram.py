import os
import h5py
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import librosa

hd5File_path = 'DataHandler/maestro-v3.0.0-hdf5.hdf5'
maestro_dataset_path = os.path.join('..', 'maestro-v3.0.0')

def GetWavefile(path):
    '''
    Get the wave file from the path
    '''
    wavefile = os.path.join(maestro_dataset_path, path)
    return wavefile

def WavToSpectrogram(wavefile):
    '''
    Convert the wave file to log mel spectrogram
    '''
    


'''
    sample_rate, samples = wavfile.read(wavefile)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
'''

    


#read hdf5 file and retrive wav path
with h5py.File(hd5File_path, 'r') as f:
    for year in f.keys():
        print(year)
        for data in f[year]:
            print(data)
            print(f[year][data]['audio_path'][()])
            pathToWavFile = f[year][data]['audio_path'][()]
            #retrieve the wave file
            wavefile = GetWavefile(pathToWavFile)

            #convert wave file to spectrogram
            spectrogram = WavToSpectrogram(wavefile)

            #add spectrogram to hdf5 file
            f[year][data]['spectrogram'] = spectrogram
