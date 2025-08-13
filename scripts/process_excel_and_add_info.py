import pandas as pd
import os
import glob
import re
from pathlib import Path
import argparse
from typing import Tuple, Optional, List

def extract_mask_and_rate(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract mask type and undersampling rate from filename.
    
    Example: 'cine_lax_2ch_kus_ktRadial16.mat' -> ('ktRadial', 16)
    The pattern is: <base_name>_<mask_type><undersampling_rate>.mat
    """
    # Remove the .mat extension
    name_without_ext = filename.replace('.mat', '')
    
    # Split by underscore and get the last component
    parts = name_without_ext.split('_')
    if len(parts) < 2:
        return None, None
    
    last_part = parts[-1]
    
    # Use regex to separate mask type and undersampling rate
    # Pattern: letters followed by digits
    match = re.match(r'^([a-zA-Z]+)(\d+)$', last_part)
    if match:
        mask_type = match.group(1)
        undersampling_rate_str = match.group(2)
        try:
            undersampling_rate = int(undersampling_rate_str)
            return mask_type, undersampling_rate
        except ValueError:
            return mask_type, None
    
    return None, None

def find_matching_file(base_dir: str, modality: str, center: str, vendor: str, patient: str, file_prefix: str) -> Optional[str]:
    """
    Find the matching .mat file in the directory structure.
    
    Directory structure: <base_dir>/<Modality>/ValidationSet/UnderSample_TaskR2/<Center>/<Vendor>/<Patient>/<File*.mat>
    
    Special handling for T1map vs T1mappost distinction:
    - When looking for "T1map", exclude files containing "T1mappost"
    - When looking for "T1mappost", look for exact "T1mappost" prefix
    """
    search_pattern = os.path.join(
        base_dir,
        modality,
        "ValidationSet",
        "UnderSample_TaskR2", 
        center,
        vendor,
        patient,
        f"{file_prefix}*.mat"
    )
    
    matching_files = glob.glob(search_pattern)
    
    # Special handling for T1map vs T1mappost distinction
    if file_prefix.lower() == "t1map":
        # When looking for "T1map", exclude files that contain "T1mappost"
        matching_files = [f for f in matching_files if "t1mappost" not in os.path.basename(f).lower()]
    elif file_prefix.lower() == "t1mappost":
        # When looking for "T1mappost", only include files that contain "t1mappost"
        matching_files = [f for f in matching_files if "t1mappost" in os.path.basename(f).lower()]
    
    if len(matching_files) == 1:
        return matching_files[0]
    elif len(matching_files) > 1:
        print(f"Warning: Multiple files found for pattern {search_pattern}")
        print(f"After filtering: {matching_files}")
        return matching_files[0]  # Return the first match
    else:
        print(f"Warning: No file found for pattern {search_pattern}")
        return None

def process_excel_file(input_excel_path: str, base_dir: str, output_excel_path: str):
    """
    Process the Excel file and add mask type and undersampling rate columns.
    """
    # Read the Excel file
    try:
        df = pd.read_excel(input_excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
    # Check if required columns exist
    required_columns = ['Modality', 'Center', 'Vendor', 'Patient', 'File']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Initialize new columns - insert Mask_Type and Undersampling_Rate before Comments if it exists
    columns = list(df.columns)
    
    # Find the position to insert the new columns (before Comments if it exists)
    if 'Comments' in columns:
        comments_idx = columns.index('Comments')
        # Insert new columns before Comments
        columns.insert(comments_idx, 'Mask_Type')
        columns.insert(comments_idx + 1, 'Undersampling_Rate')
        # Add other columns at the end
        columns.extend(['Found_File_Path', 'File_Found'])
    else:
        # If no Comments column, add new columns at the end
        columns.extend(['Mask_Type', 'Undersampling_Rate', 'Found_File_Path', 'File_Found'])
    
    # Reorder DataFrame with new column order and initialize new columns
    for col in ['Mask_Type', 'Undersampling_Rate', 'Found_File_Path', 'File_Found']:
        if col not in df.columns:
            df[col] = None
    
    df = df.reindex(columns=columns)
    
    # Set default values for File_Found
    df['File_Found'] = False
    
    # Process each row
    for idx, row in df.iterrows():
        modality = str(row['Modality'])
        center = str(row['Center'])
        vendor = str(row['Vendor'])
        patient = str(row['Patient'])
        file_prefix = str(row['File'])
        
        # Special handling: If Modality is mapping and center starts with UIH, override vendor
        if modality.lower() == 'mapping' and vendor.startswith('UIH') and not center == 'Center005':
            vendor = 'UIH_30T_umr780'
            print(f"Row {idx + 1}: Overriding vendor to 'UIH_30T_umr780' for mapping modality with UIH center")
        
        # Find the matching file
        found_file = find_matching_file(base_dir, modality, center, vendor, patient, file_prefix)
        
        if found_file:
            df.at[idx, 'Found_File_Path'] = found_file
            df.at[idx, 'File_Found'] = True
            
            # Extract filename from path
            filename = os.path.basename(found_file)
            
            # Extract mask type and undersampling rate
            mask_type, undersampling_rate = extract_mask_and_rate(filename)
            
            df.at[idx, 'Mask_Type'] = mask_type
            df.at[idx, 'Undersampling_Rate'] = undersampling_rate  # Now an integer or None
            
            print(f"Row {idx + 1}: Found {filename} -> Mask: {mask_type}, Rate: {undersampling_rate}")
        else:
            df.at[idx, 'File_Found'] = False
            print(f"Row {idx + 1}: No file found for {modality}/{center}/{vendor}/{patient}/{file_prefix}")
    
    # Save the updated DataFrame to a new Excel file
    try:
        df.to_excel(output_excel_path, index=False)
        print(f"\nResults saved to: {output_excel_path}")
        
        # Print summary
        total_rows = len(df)
        found_files = df['File_Found'].sum()
        print(f"\nSummary:")
        print(f"Total rows processed: {total_rows}")
        print(f"Files found: {found_files}")
        print(f"Files not found: {total_rows - found_files}")
        
        # Show distribution of mask types and rates
        if found_files > 0:
            print(f"\nMask Type distribution:")
            mask_counts = df['Mask_Type'].value_counts()
            for mask_type, count in mask_counts.items():
                if pd.notna(mask_type):
                    print(f"  {mask_type}: {count}")
            
            print(f"\nUndersampling Rate distribution:")
            rate_counts = df['Undersampling_Rate'].value_counts()
            for rate, count in rate_counts.items():
                if pd.notna(rate):
                    print(f"  {rate}: {count}")
        
    except Exception as e:
        print(f"Error saving Excel file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Process Excel file and find corresponding .mat files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_excel_and_add_info.py --input data.xlsx --base-dir /path/to/data --output results.xlsx
  python process_excel_and_add_info.py --input data.xlsx --base-dir /path/to/data --output results.xlsx --dry-run
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input Excel file"
    )
    
    parser.add_argument(
        "--base-dir", "-b",
        required=True,
        help="Base directory containing the .mat files"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output Excel file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually processing"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.input):
        print(f"Error: Input Excel file does not exist: {args.input}")
        return
    
    if not os.path.isdir(args.base_dir):
        print(f"Error: Base directory does not exist: {args.base_dir}")
        return
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        print(f"Would process: {args.input}")
        print(f"Base directory: {args.base_dir}")
        print(f"Would save to: {args.output}")
        
        # Read and show a sample of the Excel file
        try:
            df = pd.read_excel(args.input)
            print(f"\nInput file has {len(df)} rows and columns: {list(df.columns)}")
            print("\nFirst few rows:")
            print(df.head())
        except Exception as e:
            print(f"Error reading input file: {e}")
        return
    
    print(f"Processing Excel file: {args.input}")
    print(f"Base directory: {args.base_dir}")
    print(f"Output file: {args.output}")
    
    process_excel_file(args.input, args.base_dir, args.output)

if __name__ == "__main__":
    main()