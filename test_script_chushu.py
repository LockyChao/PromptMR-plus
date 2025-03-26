from pathlib import Path
from data import CmrxReconInferenceSliceDataset

# Instantiate the dataset with your configuration.
dataset = CmrxReconInferenceSliceDataset(
    root=Path('/common/lidxxlab/cmrchallenge/data/CMR2024/ChallengeData/MultiCoil'),
    challenge='multicoil',
    raw_sample_filter=lambda x: True,  # No filtering; adjust as needed
    transform=None,                    # Provide a transform if necessary
    num_adj_slices=5
)

print("Dataset length:", len(dataset))
#print("Volume paths:", dataset.volume_paths)

from pathlib import Path

def get_mask_path(volume_path: str, year: int) -> str:
    """
    Generate the mask path based on the volume file path and the year.
    
    Args:
        volume_path (str): The file path to the k-space volume (a .mat file).
        year (int): The year of the dataset (either 2023 or 2024).
        
    Returns:
        str: The computed mask file path.
    """
    if year == 2023:
        mask_path = volume_path.replace('.mat', '_mask.mat')
    elif year == 2024:
        mask_path = volume_path.replace('UnderSample_Task', 'Mask_Task').replace('_kus_', '_mask_')
    else:
        raise ValueError("Unsupported year: choose 2023 or 2024")
    return mask_path

# Example usage:
# Suppose you have a volume file path from your dataset.
# For demonstration, set a sample path (update this path to one of your actual files).
sample_volume_path = '/common/lidxxlab/cmrchallenge/data/CMR2024/ChallengeData/MultiCoil'

# Determine the year from your dataset root (here, using 2024 as an example)
year = 2024

# Compute the mask path using the function above
computed_mask_path = get_mask_path(sample_volume_path, year)

print("Volume path: ", sample_volume_path)
print("Computed mask path: ", computed_mask_path)

from pathlib import Path

# Create a dummy filter that passes all files (adjust as needed)
raw_sample_filter = lambda x: True

# Instantiate the dataset
data_path = Path("/common/lidxxlab/cmrchallenge/data/CMR2024/ChallengeData/MultiCoil")
dataset = CmrxReconInferenceSliceDataset(
    root=data_path,
    challenge="multicoil",
    raw_sample_filter=raw_sample_filter,
    transform=None,
    num_adj_slices=5
)

# Option 1: Retrieve a sample via __getitem__ (which loads mask as part of the sample)
sample = dataset[0]
kspace, mask, _, attrs, _path, slice_idx, num_t, num_z = sample
print("Loaded mask shape from __getitem__:", mask.shape)

# Option 2: Directly call _load_volume for a chosen volume
volume_path = dataset.volume_paths[0]
print("Volume path:", volume_path)
_, mask_direct, _ = dataset._load_volume(volume_path)
print("Directly loaded mask shape:", mask_direct.shape)

#check all the mask shape
for vp in dataset.volume_paths:
    if 'UnderSample_Task1' not in vp:
        print("Testing volume:", vp)
        _, mask_direct, _ = dataset._load_volume(vp)
        print("Mask shape for non-Task1:", mask_direct.shape)
        break
print("Checking mask shapes for volume files...")

# Loop over the first 20 volume files (adjust the range as needed)
for i, volume_path in enumerate(dataset.volume_paths[:20]):
    print(f"\nVolume {i+1}: {volume_path}")
    try:
        # Directly load the volume, which also loads the mask via _load_volume
        _, mask, _ = dataset._load_volume(volume_path)
        print("Loaded mask shape:", mask.shape)
    except Exception as e:
        print("Error loading mask:", e)


print("Checking mask shapes for UnderSample_Task2 volumes...")

# Filter volume paths for UnderSample_Task2 files
task2_paths = [vp for vp in dataset.volume_paths if 'UnderSample_Task2' in vp]
print(f"Found {len(task2_paths)} UnderSample_Task2 volume files.")

for i, volume_path in enumerate(task2_paths):
    print(f"\nVolume {i+1}: {volume_path}")
    try:
        # Load the volume (which includes the mask) for this file
        _, mask, _ = dataset._load_volume(volume_path)
        print("Loaded mask shape:", mask.shape)
    except Exception as e:
        print("Error loading mask:", e)
