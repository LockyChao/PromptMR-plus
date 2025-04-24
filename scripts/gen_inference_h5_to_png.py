#!/usr/bin/env python3

import os
import h5py
import numpy as np
from PIL import Image


def process_file(h5_path: str, out_dir: str):
    """
    Read the 'reconstruction' dataset from an H5 file and save each (Nt, Nz) slice as a PNG.

    Args:
        h5_path: Path to the input .h5 file.
        out_dir: Directory to save PNG images for this file.
    """
    with h5py.File(h5_path, 'r') as f:
        # Expecting a dataset named 'reconstruction' with shape (Nt, Nz, Nx, Ny)
        rec = f['reconstruction'][()]

    Nt, Nz, Nx, Ny = rec.shape
    base = os.path.splitext(os.path.basename(h5_path))[0]
    target_dir = os.path.join(out_dir, base)
    os.makedirs(target_dir, exist_ok=True)

    for t in range(Nt):
        for z in range(Nz):
            slice_2d = rec[t, z, :, :]
            # Normalize slice to 0-255
            minv, maxv = slice_2d.min(), slice_2d.max()
            if maxv > minv:
                norm = (slice_2d - minv) / (maxv - minv)
            else:
                norm = np.zeros_like(slice_2d)

            img_uint8 = (norm * 255).astype(np.uint8)
            img = Image.fromarray(img_uint8)

            filename = f"{base}_t{t:03d}_z{z:03d}.png"
            img.save(os.path.join(target_dir, filename))


def main(input_dir: str, output_dir: str):
    """
    Walk through the input directory, process each .h5 file, and dump PNGs into the output directory.

    Args:
        input_dir: Directory containing .h5 files.
        output_dir: Base directory to store generated PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith('.h5'):
            h5_path = os.path.join(input_dir, fname)
            process_file(h5_path, output_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert H5 reconstructions (Nt, Nz, Nx, Ny) to PNG images'
    )
    parser.add_argument(
        'input_dir', type=str,
        help='Directory containing .h5 files'
    )
    parser.add_argument(
        'output_dir', type=str,
        help='Directory to save PNG files'
    )

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
