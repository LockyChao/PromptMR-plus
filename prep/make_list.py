import os
import csv
from pathlib import Path

def collect_dataset_info(flat_dir: Path):
    file_info = []
    for file_path in sorted(flat_dir.rglob("*")):
        try:
            resolved_path = file_path.resolve(strict=True)
            if resolved_path.is_file():
                file_info.append({
                    "filename": file_path.name,
                    "symlink_path": str(file_path),
                    "real_path": str(resolved_path),
                    "extension": file_path.suffix
                })
        except FileNotFoundError:
            print(f"Warning: Broken symlink skipped: {file_path}")
    return file_info

def write_csv(data, out_file):
    if not data:
        print(f"[!] No data to write to {out_file}")
        return
    with open(out_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

# Set your dataset root directory
dataset_root = Path("/common/lidxxlab/cmrchallenge/data/CMR2025/Processed")

# Iterate and collect
train_info = collect_dataset_info(dataset_root / "train")
val_info   = collect_dataset_info(dataset_root / "val")

# Save to CSV
write_csv(train_info, "train_list.csv")
write_csv(val_info,   "val_list.csv")

print("CSV files saved.")