#!/usr/bin/env python3
"""
Evaluation script: load predicted and ground-truth HDF5 (.h5) and mask NIfTI (.nii.gz),
apply broadcast mask, compute PSNR and SSIM, and write results to CSV.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
import nibabel as nib
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_h5_data(path: Path) -> np.ndarray:
    """Load the 'reconstruction_rss' dataset (or fallback to first dataset) from an HDF5 (.h5) file."""
    with h5py.File(path, 'r') as f:
        # Prefer 'reconstruction_rss' if present
        if 'reconstruction_rss' in f:
            data = f['reconstruction_rss'][()]
        else:
            # Fallback: root-level datasets
            data = None
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data = f[key][()]
                    break
            # Recursive fallback
            if data is None:
                def _visitor(name, obj):
                    nonlocal data
                    if data is None and isinstance(obj, h5py.Dataset):
                        data = obj[()]
                f.visititems(_visitor)
            if data is None:
                raise ValueError(f"No dataset found in H5 file: {path}")
    return data

def align_h5(data: np.ndarray) -> np.ndarray:
    """Reorder H5 data to (x, y, z) or (x, y, z, t) to match NIfTI shape."""
    if data.ndim == 3:
        # from (z, x, y) -> (x, y, z)
        return np.transpose(data, (1, 2, 0))
    if data.ndim == 4:
        # from (z, t, x, y) -> (x, y, z, t)
        return np.transpose(data, (2, 3, 0, 1))
    raise ValueError(f"Unsupported H5 data dimensions: {data.shape}")

def load_and_align_h5(path: Path) -> np.ndarray:
    """Load and align H5 file to numpy array matching NIfTI orientation."""
    raw = load_h5_data(path)
    return align_h5(raw)

def load_mask(path: Path) -> np.ndarray:
    """Load a NIfTI file and return its data array."""
    nii = nib.load(str(path))
    return nii.get_fdata()

def broadcast_mask(mask: np.ndarray) -> np.ndarray:
    """Broadcast the first slice mask across all slices along the z-axis."""
    if mask.ndim == 3:
        base = mask[:, :, 0] > 0
        return np.broadcast_to(base[:, :, None], mask.shape)
    if mask.ndim == 4:
        # mask shape (x, y, z, t) -> use first z-slice across all z
        base = mask[:, :, 0, :] > 0  # shape (x, y, t)
        return np.broadcast_to(base[:, :, None, :], mask.shape)
    raise ValueError(f"Unsupported mask dimensions: {mask.shape}")

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Compute mean PSNR and SSIM between gt and pred arrays."""
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch gt {gt.shape} vs pred {pred.shape}")
    dims = gt.ndim
    psnrs, ssims = [], []
    if dims == 3:
        # iterate slices along z-axis
        for k in range(gt.shape[2]):
            g = gt[:, :, k]
            p = pred[:, :, k]
            if np.all(g == 0):
                continue
            maxval = g.max()
            psnrs.append(peak_signal_noise_ratio(g, p, data_range=maxval))
            ssims.append(structural_similarity(g, p, data_range=maxval))
    elif dims == 4:
        # iterate time and slice dimensions
        for t in range(gt.shape[3]):
            for k in range(gt.shape[2]):
                g = gt[:, :, k, t]
                p = pred[:, :, k, t]
                if np.all(g == 0):
                    continue
                maxval = g.max()
                psnrs.append(peak_signal_noise_ratio(g, p, data_range=maxval))
                ssims.append(structural_similarity(g, p, data_range=maxval))
    else:
        raise ValueError(f"Unsupported data dimensions: {gt.shape}")
    return {'psnr': float(np.mean(psnrs)) if psnrs else np.nan,
            'ssim': float(np.mean(ssims)) if ssims else np.nan}

def main(pred_dir: Path, gt_dir: Path, mask_dir: Path, output_csv: Path):
    subjects = []
    records = []
    for pred_file in sorted(pred_dir.glob('*.h5')):
        subj = pred_file.stem
        subj_clean = subj.replace('_v73', '')
        gt_file = gt_dir / f"{subj}.h5"
        mask_file = mask_dir / subj_clean / "mask.nii.gz"
        if not gt_file.exists() or not mask_file.exists():
            print(f"Skipping subject {subj}: missing GT or mask")
            continue
        print(f"Processing subject {subj}...")
        gt = load_and_align_h5(gt_file)
        pred = load_and_align_h5(pred_file)
        mask = load_mask(mask_file)       
        #omit all axis with shape 1
        mask_b = np.squeeze(mask)
        #mask_b axis shift to (2, 0, 1)
        mask_b = np.transpose(mask_b, (2, 0, 1))
        mask_b = broadcast_mask(mask_b)
        gt = np.squeeze(gt)
        pred = np.squeeze(pred)
        gt_masked = gt * mask_b
        pred_masked = pred * mask_b
        #print shape of gt_masked and pred_masked
        print(f"gt_masked shape: {gt_masked.shape}, pred_masked shape: {pred_masked.shape}")
        m = compute_metrics(gt_masked, pred_masked)
        m['subject'] = subj
        records.append(m)
    # save
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")
    print(f"Mean PSNR: {df['psnr'].mean():.4f}")
    print(f"Mean SSIM: {df['ssim'].mean():.4f}")

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='Evaluate reconstructions against GT with mask')
    parser.add_argument('--pred_dir', type=Path, help='Directory of .h5 prediction files', default=Path("/common/lidxxlab/chaowei/data/CMR2025/DirectInference_model2025/reconstructions"))
    parser.add_argument('--gt_dir',   type=Path, help='Directory of .h5 ground-truth files', default=Path("/common/lidxxlab/cmrchallenge/data/CMR2025/Processed/MultiCoil"))
    parser.add_argument('--mask_dir', type=Path, help='Directory of .nii.gz mask files', default=Path("/common/lidxxlab/Yifan/PromptMR-plus/full_nii_train"))
    parser.add_argument('--output_csv', type=Path, default=Path('/common/lidxxlab/Yifan/PromptMR-plus/prep/metrics_fardad.csv'),
                        help='Output CSV filename')
    args = parser.parse_args()
    '''
    gt_dir = Path("/common/lidxxlab/cmrchallenge/data/CMR2025_fardad/processed_h5ii/val")
    pred_dir = Path("/common/lidxxlab/cmrchallenge/code/Inference/predict/fardad_new/reconstructions")
    mask_dir = Path("/common/lidxxlab/Yifan/PromptMR-plus/fardad_val")
    output_csv = Path("/common/lidxxlab/Yifan/PromptMR-plus/prep/metrics_fardad.csv")
    main(pred_dir, gt_dir, mask_dir, output_csv)
    '''

    main(gt_dir, pred_dir)
    '''
