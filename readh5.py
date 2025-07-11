#!/usr/bin/env python3
"""
Script to analyze and display information about an H5 file structure and contents
"""
import os
import h5py
import numpy as np
from pathlib import Path
# File path to analyze
#file_path = "/common/lidxxlab/cmrchallenge/data/CMR2025/Processed/MultiCoil/Center001_UIH_30T_umr780_P001_cine_lax_3ch.h5"
#file_path = "/common/lidxxlab/cmrchallenge/data/CMR2025/Processed/MultiCoil/Center001_UIH_30T_umr780_P001_T2w.h5"
file_path = "/common/lidxxlab/cmrchallenge/data/CMR2025/ChallengeData/MultiCoil/T1w/TrainingSet/FullSample/Center003/UIH_30T_umr880/P004/T1w.mat"
def format_bytes(size):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"
def analyze_h5_file(filepath):
    """Analyze and print information about an H5 file"""
    print(f"Analyzing: {filepath}")
    # Get file size
    file_size = os.path.getsize(filepath)
    print(f"File size: {format_bytes(file_size)}")
    try:
        with h5py.File(filepath, 'r') as f:
            # Print structure info
            print("\n--- File Structure ---")
            def print_attrs(name, obj):
                """Print attributes of a dataset/group"""
                print(f"\n{name}:")
                if isinstance(obj, h5py.Dataset):
                    print(f"  Type: Dataset")
                    print(f"  Shape: {obj.shape}")
                    print(f"  Datatype: {obj.dtype}")
                    print(f"  Size: {format_bytes(obj.size * obj.dtype.itemsize)}")
                    # Show statistics for numerical data if not too large
                    if obj.size > 0 and obj.size < 10**8 and np.issubdtype(obj.dtype, np.number):
                        try:
                            data = obj[()]
                            print(f"  Min: {np.min(data)}")
                            print(f"  Max: {np.max(data)}")
                            print(f"  Mean: {np.mean(data)}")
                            print(f"  Std: {np.std(data)}")
                        except Exception as e:
                            print(f"  Could not compute statistics: {e}")
                else:
                    print(f"  Type: Group")
                # Print attributes
                if len(obj.attrs) > 0:
                    print("  Attributes:")
                    for key, value in obj.attrs.items():
                        if len(str(value)) > 100:
                            value_str = str(value)[:97] + "..."
                        else:
                            value_str = value
                        print(f"    {key}: {value_str}")
            # Visit all groups and datasets
            f.visititems(print_attrs)
    except Exception as e:
        print(f"Error analyzing file: {e}")
if __name__ == "__main__":
    analyze_h5_file(file_path)