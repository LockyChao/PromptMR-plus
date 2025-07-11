import h5py
import numpy as np
import matplotlib.pyplot as plt

import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_hdf5_data(fname):
    with h5py.File(fname, 'r') as hf:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        hf.visititems(print_structure)

        if "data" in hf:
            mask = hf["data"][:]  # Load the actual mask data
            print(f"mask shape: {mask.shape}")

            # Visualize the first entry if it's 3D (e.g., [2, H, W])
            mask_to_show = mask[1] if mask.ndim == 3 else mask
            plt.imshow(
                mask_to_show,
                cmap='gray',
                extent=[0, mask_to_show.shape[1], 0, mask_to_show.shape[0]],
                aspect='equal'
            )
            plt.colorbar()
            plt.title("Mask Visualization")
            plt.xlabel("Width")
            plt.ylabel("Height")

            # Save and display the figure
            mask_fname = '/common/lidxxlab/Yifan/PromptMR-plus/utils/mask_1.png'
            plt.savefig(mask_fname)
            plt.show()
            print(f"Mask saved to {mask_fname}")
        else:
            print("No 'data' dataset found in HDF5.")
            
fname = '/common/lidxxlab/cmrchallenge/data/CMR2025/Processed/Mask/acc24_260x512.h5'
read_hdf5_data(fname)