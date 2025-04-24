#!/usr/bin/env python3
import os
import glob
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(description="Copy and rename PNG files based on their folder structure.")
    parser.add_argument('--input_folder', type=str, required=True,
                        help="Input folder, e.g., folder1")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Output folder, e.g., folder2")
    args = parser.parse_args()
    
    # Build the glob pattern: folder1/*/TrainingSet/ImageShow/*/*/*/magnitude/*.png
    pattern = os.path.join(args.input_folder, "*/TrainingSet/ImageShow/*/*/*/magnitude", "*.png")
    
    png_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(png_files)} PNG files.")

    # Ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    for file_path in png_files:
        # Split the file path into parts using the OS separator
        parts = file_path.split(os.sep)
        # For a path like:
        # folder1/x/TrainingSet/ImageShow/A/B/C/magnitude/file.png
        # parts[-4] corresponds to 'B' and parts[-3] corresponds to 'C'
        a = parts[-4]  # EDIT: using second last directory before "magnitude"
        b = parts[-3]  # EDIT: using last directory before "magnitude"
        filename = os.path.basename(file_path)
        
        # Construct new filename: a_b_filename.png
        new_filename = f"{a}_{b}_{filename}"
        dst = os.path.join(args.output_folder, new_filename)
        
        shutil.copy(file_path, dst)
        print(f"Copied {file_path} to {dst}")

if __name__ == "__main__":
    main()
