"""
Description: This script is the main entry point for the LightningCLI.
"""
import os
import sys
from itertools import chain
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
import yaml
import torch
import numpy as np

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import BasePredictionWriter
from types import SimpleNamespace

from mri_utils import save_reconstructions
from pl_modules import PromptMrModule


def preprocess_save_dir():
    """Ensure `save_dir` exists, handling both command-line arguments and YAML configuration."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, nargs="*",
                        help="Path(s) to YAML config file(s)")
    parser.add_argument("--trainer.logger.save_dir",
                        type=str, help="Logger save directory")
    args, _ = parser.parse_known_args(sys.argv[1:])

    save_dir = None  # Default to None

    if args.config:
        for config_path in args.config:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding='utf-8') as f:
                    try:
                        config = yaml.safe_load(f)
                        if config is not None:
                            # Safely navigate to trainer.logger.save_dir
                            trainer = config.get("trainer", {})
                            logger = trainer.get("logger", {})
                            if isinstance(logger, dict) :  # Ensure logger is a dictionary
                                yaml_save_dir = logger.get(
                                    "init_args", {}).get("save_dir")
                                if yaml_save_dir:
                                    save_dir = yaml_save_dir  # Use the first valid save_dir found
                                    break
                    except yaml.YAMLError as e:
                        print(f"Error parsing YAML file {config_path}: {e}")

    for i, arg in enumerate(sys.argv):
        if arg == "--trainer.logger.save_dir":
            save_dir = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
            break

    if not save_dir:
        print("Logger save_dir is None. No action taken.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Pre-created logger save_dir: {save_dir}")


class CustomSaveConfigCallback(SaveConfigCallback):
    '''save the config file to the logger's run directory, merge tags from different configs'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merged_tags = self._collect_tags_from_configs()

    def _collect_tags_from_configs(self):
        config_files = []
        merged_tags = set()

        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_files.append(sys.argv[i + 1])

        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if isinstance(config_data, dict):
                            logger = config_data.get('trainer', {}).get(
                                'logger', {})
                            if logger and isinstance(logger, dict):
                                tags = logger.get('init_args', {}).get('tags', [])
                                if isinstance(tags, list):
                                    merged_tags.update(tags)
                except (yaml.YAMLError, IOError) as e:
                    print(f"Warning: Error reading {config_file}: {str(e)}")
        return merged_tags

    def setup(self, trainer, pl_module, stage):
        if hasattr(self.config, 'trainer') and hasattr(self.config.trainer, 'logger'):
            logger_config = self.config.trainer.logger
            if hasattr(logger_config, 'init_args'):
                logger_config.init_args['tags'] = list(self.merged_tags)
                if hasattr(trainer, 'logger') and trainer.logger is not None:
                    trainer.logger.experiment.tags = list(self.merged_tags)

        super().setup(trainer, pl_module, stage)

    def save_config(self, trainer, pl_module, stage) -> None:
        """Save the configuration file under the logger's run directory."""
        if stage == "predict":
            print("Skipping saving configuration in predict mode.")
            return  
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            project_name = trainer.logger.experiment.project_name()
            run_id = trainer.logger.experiment.id
            save_dir = trainer.logger.save_dir
            run_dir = os.path.join(save_dir, project_name, run_id)
            
            os.makedirs(run_dir, exist_ok=True)
            config_path = os.path.join(run_dir, "config.yaml")
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            print(f"Configuration saved to {config_path}")


class CustomWriter(BasePredictionWriter):
    """
    A custom prediction writer to save reconstructions to disk.
    """

    def __init__(self, output_dir: Path, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.outputs = defaultdict(list)
        
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        """
        Collect predictions batch by batch and organize them by volume.
        Assumes `predictions` contains a dictionary with 'volume_id' and 'slice_prediction'.
        """
        pass
    
    # In main.py
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # --- This distributed gathering logic is from your code ---
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, predictions)
            if not trainer.is_global_zero:
                return
            # This correctly flattens the list of lists from all GPUs
            predictions = [item for sublist in gathered for item in sublist]
        else:
            # single-GPU / no DDP: keep predictions as a flat list of dicts
            # (if it ever comes in wrapped as a list-of-lists, unwrap one level)
            if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
                predictions = predictions[0]

        # --- Grouping logic from your code, slightly modified ---
        outputs = defaultdict(list)

        for batch_output in predictions:
            # Loop through items in the batch (handles batch_size > 1)
            for i in range(len(batch_output["fname"])):
                fname = batch_output["fname"][i]
                # Ensure fname is a string, not a tuple
                if isinstance(fname, tuple):
                    fname = fname[0]
                
                # Store the entire dictionary for this slice
                outputs[fname].append({
                    "slice_num": batch_output["slice_num"][i],
                    "time_frame": batch_output["time_frame"][i],
                    "num_slc": batch_output["num_slc"][i],
                    "output": batch_output["output"][i],
                    "has_fake_time_dim": batch_output["has_fake_time_dim"][i]
                })
                
        # Directory for saving the final volumes
        save_dir = self.output_dir / "TaskS2/MultiCoil"
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- FIX #1: This entire block replaces your old "Sort and Stack" logic ---
        # Process each file independently to prevent metadata mix-ups
        for fname, slice_data_list in outputs.items():
            if not slice_data_list:
                continue

            try:
                # a. Sort all slices for THIS file by time, then by slice number
                sorted_slices = sorted(
                    slice_data_list,
                    key=lambda s: (s['time_frame'].item(), s['slice_num'].item())
                )

                # b. Stack all sorted slices into one tensor
                stacked_volume = torch.stack([s['output'].squeeze() for s in sorted_slices])

                # c. Get the correct number of slices for THIS file from its own data
                num_slices_per_time = sorted_slices[0]['num_slc'].item()
                num_time_frames = len(sorted_slices) // num_slices_per_time
                has_fake_time_dim = sorted_slices[0]['has_fake_time_dim']
                
                # d. Final check to ensure data is consistent
                if len(sorted_slices) % num_slices_per_time != 0:
                    print(f"Warning: Inconsistent data for {fname}. Skipping.")
                    continue
                
                h, w = stacked_volume.shape[-2], stacked_volume.shape[-1]
                
                # e. Reshape into the final 4D volume using the CORRECT dimensions
                final_4d_volume = stacked_volume.view(num_time_frames, num_slices_per_time, h, w)

                # f. Check if we need to remove fake time dimension 
                # Env toggle: save real MATLAB .mat files if requested
                # 
                
                save_as_mat = True

                if has_fake_time_dim and num_time_frames == 2:
                    # Original data was 4D, fake time dimension was added, remove it
                    print(f"Removing fake time dimension for {fname}: {final_4d_volume.shape} -> 3D")
                    # Take the first time frame and remove the time dimension
                    final_volume = final_4d_volume[0]  # Shape: (num_slices_per_time, h, w)
                    print(f"Saving 3D volume for {fname} with shape {final_volume.shape}")
                    save_reconstructions(final_volume, fname, save_dir, is_mat=save_as_mat)
                else:
                    print(f"Saving 4D volume for {fname} with shape {final_4d_volume.shape}")
                    # The save function is now much simpler
                    save_reconstructions(final_4d_volume, fname, save_dir, is_mat=save_as_mat)

            except Exception as e:
                print(f"CRITICAL ERROR while processing/saving {fname}. Error: {e}")
                
        print(f"Done! Reconstructions saved to {save_dir}")


class CustomLightningCLI(LightningCLI):

    def instantiate_classes(self):
        super().instantiate_classes()
        if self.config_init.subcommand == 'predict':
            ckpt_path_to_load = self.config_init.predict.ckpt_path
            print(f"\n✅ LOADING CHECKPOINT FROM: {ckpt_path_to_load}\n")
            # ✂️ DON’T call load_from_checkpoint here!
            # self.model = PromptMrModule.load_from_checkpoint(self.config_init.predict.ckpt_path)


def run_cli():
    preprocess_save_dir()

    cli = CustomLightningCLI(
        save_config_callback=None,
        save_config_kwargs={"overwrite": True},
        run=True  # Let Lightning handle predict()
    )

    
if __name__ == "__main__":
    run_cli()

