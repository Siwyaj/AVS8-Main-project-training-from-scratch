'''
This file is just to confirm the structure of the hdf5 file
'''

def ReadHDF5Structure(hdf5_path):
    import h5py
    import os

    # Check if the file exists
    if not os.path.exists(hdf5_path):
        print(f"File {hdf5_path} does not exist.")
        return

    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        # Print the structure of the HDF5 file
        def print_structure(name, obj):
            print(f"{name}: {obj}")

        f.visititems(print_structure)