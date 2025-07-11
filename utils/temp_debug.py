#!/usr/bin/env python3
"""
Convert all MATLAB .mat files (v7.3/HDF5 and legacy) in a directory to MATLAB v5 legacy format,
preserving directory structure and extracting only variables containing 'reconstruction'.
"""
import sys
from pathlib import Path
import scipy.io
import h5py
import numpy as np

def run4Ranking_2025(img: np.ndarray, filetype: str) -> np.ndarray:
    """
    Translate MATLAB run4Ranking_2025 to Python: select central slices/timeframes,
    and crop the central region (1/6 original area) for ranking.
    """
    # Swap axes before cropping: for 4D (sz,t,sy,sx) and for 3D (sy,sz,sx)
    if img.ndim == 3:
        # Original shape (sx, sy, sz) → new shape (sy, sz, sx)
        data = img.transpose((1, 2, 0))
        sx, sy, sz = data.shape
        t = 1
    elif img.ndim == 4:
        # Original shape (sx, sy, sz, t) → new shape (sz, t, sy, sx)
        data = img.transpose((2, 3, 1, 0))
        sx, sy, sz, t = data.shape
    else:
        raise ValueError(f"Unexpected image dimensions: {img.shape}")

    ft = filetype.lower()
    isBlackBlood = any(x in ft for x in ['blackblood', 't1w', 't2w'])
    detectMap = ['t1map', 't2map', 't2smap', 't1mappost']
    isMapping = any(x in ft for x in detectMap)
    isT1rho = 't1rho' in ft

    # Clip slices according to MATLAB logic (1-based to 0-based conversion)
    if sz < 3:
        sliceToUse = list(range(sz))
    else:
        matlab_center = int(round(sz / 2))
        # MATLAB uses slices [center-1, center]; convert to zero-based indices
        sliceToUse = [matlab_center - 2, matlab_center - 1]

    # Select time frames
    if isBlackBlood or t == 1:
        timeFrameToUse = [0]
    elif isMapping or isT1rho:
        timeFrameToUse = list(range(t))
    else:
        timeFrameToUse = list(range(min(3, t)))

    # Extract selected slices and frames
    if data.ndim == 3:
        selected = data[:, :, sliceToUse]
        selected = selected[:, :, :, np.newaxis]
    else:
        selected = data[:, :, sliceToUse, :]
        selected = selected[:, :, :, timeFrameToUse]

    # Crop central region: height sx/3, width sy/2
    '''
    crop_h = round(sx / 3)
    crop_w = round(sy / 2)
    start_h = (sx - crop_h) // 2
    start_w = (sy - crop_w) // 2
    selected = np.abs(selected)
    cropped = selected[start_h:start_h + crop_h, start_w:start_w + crop_w]
    '''
    # Cast to single precision to match MATLAB's 'single'
    return selected.astype(np.single)

def convert_mat_to_legacy(input_dir: str, output_dir: str, cons_i: int, batch_size: int = 20):
    """
    Recursively find .mat and .h5 files in input_dir, extract datasets/variables
    with 'reconstruction' in their name, and save them as v5 .mat files in output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    # Gather all .mat files and sort for consistent batch splitting
    mat_files = sorted(input_path.rglob("*.mat"))
    # Determine slice for this batch
    start = cons_i * batch_size
    end = start + batch_size
    for idx, in_file in enumerate(mat_files):
        # Only process this batch's slice
        if idx < start:
            continue
        if idx >= end:
            break
        try:
            rel = in_file.relative_to(input_path)

            if h5py.is_hdf5(in_file):
                with h5py.File(in_file, 'r') as f:
                    keys = [k for k in f.keys() if 'img4ranking' in k.lower()]
                    if not keys:
                        print(f"No reconstruction dataset in {in_file}, skipping")
                        continue
                    mat_dict = {k: f[k][()] for k in keys}
            else:
                full_mat = scipy.io.loadmat(in_file)
                keys = [k for k in full_mat.keys() if 'img4ranking' in k.lower()]
                if not keys:
                    print(f"No reconstruction variable in {in_file}, skipping")
                    continue
                mat_dict = {k: full_mat[k] for k in keys}

            # Only keep cropped data
            cropped = None
            for data in mat_dict.values():
                try:
                    print(data.shape)
                    cropped = data
                    #cropped = run4Ranking_2025(data, in_file.name)
                    break
                except Exception as e:
                    print(f"Error cropping {in_file}: {e}")
            if cropped is None:
                print(f"No reconstruction data to crop in {in_file}, skipping")
                continue
            print(f"Debug: {in_file.name} cropped shape: {cropped.shape}")
            
            # Save only cropped result under original structure
            out_file = output_path / rel.with_suffix('.mat')
            out_file.parent.mkdir(parents=True, exist_ok=True)
            scipy.io.savemat(out_file, {'img4ranking': cropped}, format='5')
            print(f"Saved cropped MATLAB at {out_file}")

        except Exception as e:
            print(f"Error converting {in_file}: {e}")

def main():
    cons_i = int(sys.argv[1])
    input_dir = "/common/lidxxlab/cmrchallenge/code/Inference/predict/cmr25validation/TaskR2_eval"
    output_dir = "/common/lidxxlab/cmrchallenge/code/Inference/predict/cmr25validation/TaskR2_cropped"
    convert_mat_to_legacy(input_dir, output_dir, cons_i)

if __name__ == "__main__":
    main()
