'''
This is the main script.
Here the following can be done:
    -create hdf5 file
    -train the model
    -transcribe an audio file to midi
'''
from hdf5.Makehdf5 import MakeHDF5
from hdf5.ReadHDF5Structure import ReadHDF5Structure
from tools.dataHandlers.DataHandler import Sampler, MaestroHDF5Dataset, collate_fn
from utilities.utilities import StatisticsContainer
from utilities.evaluate import SegmentEvaluator
from model.CRNNsupermodel import CRNNModel
import os
import torch
import torch.optim as optim
import Config.Config as Config
from utilities.pytorch_utils import move_data_to_device
from utilities.loss import regress_onset_offset_frame_velocity_bce

def train_model(hdf5_path):
    
    if torch.cuda.is_available():
        print('Using GPU.')
        device = 'cuda'
    else:
        print('Using CPU.')
        device = 'cpu'    

    num_workers = 8
    

    #load the dataset
    train_dataset = MaestroHDF5Dataset(hdf5_path)

    #create the train sampler
    train_sampler = Sampler(hdf5_path)

    #dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
            batch_sampler=train_sampler, collate_fn=collate_fn, 
            num_workers=num_workers, pin_memory=True)
    

    #load the model
    model = CRNNModel()
    
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, 
    betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    print('GPU number: {}'.format(torch.cuda.device_count()))

    if 'cuda' in str(device):
        model.to(device)


    evaluator = SegmentEvaluator(model, Config.batch_size)
    statistics_container = StatisticsContainer('statistics')
    for batch_data_dict in train_loader:
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
    model.train()
    batch_output_dict = model(batch_data_dict['waveform'])

    iteration = 0
    print(iteration, loss)

    loss = regress_onset_offset_frame_velocity_bce(model, batch_output_dict, batch_data_dict)

    # Backward
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()

if __name__ == '__main__':

    a=1 #just so it is not empty
    import os
    HDF5file_path = os.path.join(os.path.dirname(__file__),"hdf5/maestro-v3.0.0.hdf5")
    print("HDF5 file path:", HDF5file_path)
    #outcomment the part you want to do
    #first create the hdf5 file
    MakeHDF5(HDF5file_path)#make hdf5 file
    ReadHDF5Structure(HDF5file_path)#to inspec the structure of the hdf5 file

    #then train the model
    #train_model(HDF5file_path)

