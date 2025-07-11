#!/usr/bin/env python3
"""
Convert all MATLAB .mat files (both v7.3/HDF5 and legacy formats) under a hard‑coded input directory into grayscale PNG images, preserving directory structure.

For multi-dimensional arrays (>2D), collapses all leading dimensions via mean to produce a single 2D slice.
"""
from pathlib import Path
import numpy as np
import scipy.io
import h5py
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from typing import Tuple, Optional
import csv
import pandas as pd

def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]
    
def process_h5_files(pred_dir, output_dir, cons_i):
    """
    Process all H5 files in the given directory:
    1. Find all .h5 files
    2. Extract 'reconstruction_rss' dataset
    3. Save all slices as PNG images
    
    Args:
        pred_dir: Directory containing H5 files
        output_dir: Directory where PNG images will be saved
        cons_i: Batch index for processing subsets of files
    """
    index = 0
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all H5 files
    h5_files = glob.glob(os.path.join(pred_dir, "**/*.h5"), recursive=True)
    
    # Also look for .mat files that might be HDF5
    mat_files = glob.glob(os.path.join(pred_dir, "**/*.mat"), recursive=True)
    potential_h5_files = [f for f in mat_files if h5py.is_hdf5(f)]
    h5_files.extend(potential_h5_files)
    
    print(f"Found {len(h5_files)} H5 files")
    
    # Process each file
    for file_path in h5_files:
        try:
            index += 1
            if (index < cons_i * 20):
                continue
            if (index >= (cons_i + 1) * 20):
                break
                
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            file_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(file_output_dir, exist_ok=True)
            
            print(f"Processing {file_path}")
            with h5py.File(file_path, 'r') as f:
                # Look for reconstruction_rss dataset
                if 'reconstruction_rss' in f:
                    data = f['reconstruction_rss'][()]
                    print(f"  Found 'reconstruction_rss', shape: {data.shape}")
                else:
                    # Look for any dataset containing "reconstruction"
                    recon_keys = [key for key in f.keys() if 'reconstruction' in key.lower()]
                    if recon_keys:
                        data = f[recon_keys[0]][()]
                        print(f"  Using '{recon_keys[0]}', shape: {data.shape}")
                    else:
                        print(f"  No reconstruction dataset found in {file_path}, skipping")
                        continue
                
                # Process based on data dimensionality
                if data.ndim == 2:
                    # Single 2D image
                    output_path = os.path.join(file_output_dir, f"{base_name}.png")
                    img_uint8 = normalize_to_uint8(data)
                    Image.fromarray(img_uint8).save(output_path)
                    print(f"  Saved single slice to {output_path}")
                
                elif data.ndim == 3:
                    # Save each slice along the first dimension
                    for i in range(data.shape[0]):
                        slice_data = data[i, :, :]
                        output_path = os.path.join(file_output_dir, f"{base_name}_slice{i:03d}.png")
                        img_uint8 = normalize_to_uint8(slice_data)
                        Image.fromarray(img_uint8).save(output_path)
                    print(f"  Saved {data.shape[0]} slices from 3D volume")
                
                elif data.ndim == 4:
                    # Save slices from 4D volume (assuming dim1 x dim2 x height x width)
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            slice_data = data[i, j, :, :]
                            output_path = os.path.join(file_output_dir, f"{base_name}_dim1_{i:03d}_dim2_{j:03d}.png")
                            img_uint8 = normalize_to_uint8(slice_data)
                            Image.fromarray(img_uint8).save(output_path)
                    print(f"  Saved {data.shape[0] * data.shape[1]} slices from 4D volume")
                
                else:
                    # For higher dimensions, extract and save 2D slices by iterating through combinations of the first dimensions
                    total_slices = 0
                    # Get the number of dimensions before the 2D image dimensions
                    leading_dims = data.shape[:-2]
                    # Generate all combinations of indices for the leading dimensions
                    indices = np.ndindex(*leading_dims)
                    for idx in indices:
                        # Use the indices to extract a 2D slice
                        # Create slicing object: idx + (slice(None), slice(None)) to select all of last two dimensions
                        slicing = idx + (slice(None), slice(None))
                        slice_data = data[slicing]
                        
                        # Create descriptive filename with all dimension indices
                        idx_str = '_'.join([f"dim{d}_{i:03d}" for d, i in enumerate(idx)])
                        output_path = os.path.join(file_output_dir, f"{base_name}_{idx_str}.png")
                        
                        img_uint8 = normalize_to_uint8(slice_data)
                        Image.fromarray(img_uint8).save(output_path)
                        total_slices += 1
                    
                    print(f"  Saved {total_slices} slices from {data.ndim}D volume")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"Finished processing {index} files")

def normalize_to_uint8(data):
    """
    Normalize the data to uint8 for saving as an image.
    """
    # Handle complex data
    if np.iscomplexobj(data):
        data = np.abs(data)
    
    # Normalize to [0, 255]
    if data.min() == data.max():
        # Handle constant image
        return np.zeros_like(data, dtype=np.uint8)
    else:
        normalized = (data - data.min()) / (data.max() - data.min()) * 255
        return normalized.astype(np.uint8)

def collapse_to_2d(data):
    """
    Collapse a multi-dimensional array to 2D by averaging over leading dimensions.
    """
    target_shape = data.shape[-2:]  # Last two dimensions
    result = data
    while result.ndim > 2:
        result = np.mean(result, axis=0)
    return result

def main():
    cons_i = int(sys.argv[1])
    pred_dir = "/common/lidxxlab/cmrchallenge/data/CMR2025/DirectInference_model2025/reconstructions"
    output_dir = "/common/lidxxlab/cmrchallenge/data/CMR2025/DirectInference_model2025/reconstructions_png"
    
    process_h5_files(pred_dir, output_dir, cons_i)

if __name__ == "__main__":
    main()
