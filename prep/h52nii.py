#!/usr/bin/env python3
"""
Convert all MATLAB .mat files (both v7.3/HDF5 and legacy formats) under a hard‑coded input directory into grayscale PNG images and NIfTI volumes, preserving directory structure.

For multi-dimensional arrays (>2D), collapses all leading dimensions via mean to produce a single 2D slice for PNGs, but saves the full volume as a .nii.gz.
"""
from pathlib import Path
import numpy as np
import scipy.io
import h5py
import nibabel as nib
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from typing import Tuple, Optional
import csv
import pandas as pd
import os
import glob
import sys

def process_h5_files(pred_dir, output_dir, cons_i):
    """
    Process all H5/.mat files in the given directory:
    1. Find all .h5 and HDF5-format .mat files
    2. Extract 'reconstruction_rss' (or fallback) dataset
    3. Save full volume as NIfTI (.nii.gz)
    4. Save 2D slices as PNG images
    """
    index = 0
    os.makedirs(output_dir, exist_ok=True)
    h5_files = glob.glob(os.path.join(pred_dir, "**/*.h5"), recursive=True)
    mat_files = glob.glob(os.path.join(pred_dir, "**/*.mat"), recursive=True)
    h5_files.extend([f for f in mat_files if h5py.is_hdf5(f)])
    print(f"Found {len(h5_files)} files to process")

    for file_path in h5_files:
        try:
            index += 1
            # batch-level slicing logic
            if index < cons_i * 20:
                continue
            if index >= (cons_i + 1) * 20:
                break

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            file_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(file_output_dir, exist_ok=True)
            print(f"Processing {file_path}")

            # read data
            with h5py.File(file_path, 'r') as f:
                if 'reconstruction_rss' in f:
                    data = f['reconstruction_rss'][()]
                else:
                    recon_keys = [k for k in f.keys() if 'reconstruction' in k.lower()]
                    if recon_keys:
                        data = f[recon_keys[0]][()]
                    else:
                        print(f"  No reconstruction dataset in {file_path}, skipping.")
                        continue

            if data.ndim == 4:
                nii_data = np.transpose(data, (2, 3, 0, 1))
            elif data.ndim == 3:
                nii_data = np.transpose(data, (2, 0, 1))
                
            # SAVE NIfTI VOLUME
            affine = np.eye(4)
            nii_img = nib.Nifti1Image(nii_data.astype(np.float32), affine)
            nii_path = os.path.join(file_output_dir, f"{base_name}.nii.gz")
            nib.save(nii_img, nii_path)
            print(f"  Saved NIfTI volume to {nii_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    print(f"Completed processing {index} files")


def normalize_to_uint8(data):
    if np.iscomplexobj(data):
        data = np.abs(data)
    if data.min() == data.max():
        return np.zeros_like(data, dtype=np.uint8)
    norm = (data - data.min()) / (data.max() - data.min()) * 255
    return norm.astype(np.uint8)

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    pdir = "/common/lidxxlab/cmrchallenge/data/CMR2025_fardad/processed_h5/val"
    odir = "/common/lidxxlab/Yifan/PromptMR-plus/fardad_val"
    process_h5_files(pdir, odir, idx)
