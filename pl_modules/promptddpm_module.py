import torch
from data import transforms
from pl_modules import MriModule
from typing import List
import copy 
from mri_utils import SSIMLoss
import torch.nn.functional as F
import importlib
from configs.ddpm_continuous import get_config
import os
import time
import logging
from models import ddpm, sde_lib
from models import losses
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
from absl import flags
import torch
from models.ddpm_utils import *
import wandb
import torchvision.utils as vutils

def get_model_class(module_name, class_name="PromptDDPM"):
    """
    Dynamically imports the specified module and retrieves the class.

    Args:
        module_name (str): The module to import (e.g., 'model.m1', 'model.m2').
        class_name (str): The class to retrieve from the module (default: 'PromptMR').

    Returns:
        type: The imported class.
    """
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

class PromptDDPMModule(MriModule):

    def __init__(self):
        super().__init__()
        self.config = config = get_config()
        self.save_hyperparameters()

        self.model = mutils.create_model(config)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=config.model.ema_rate)
        self.scaler = get_data_scaler(config)
        optimizer = losses.get_optimizer(config, self.model.parameters())
        self.state = dict(optimizer=optimizer, model=self.model, ema=self.ema, step=0)

        sde = sde_lib.VPSDE(config)
        optimize_fn = losses.optimization_manager(config)
        self.train_step_fn = losses.get_step_fn(
            config,
            sde,
            train=True,
            optimize_fn=optimize_fn,
            reduce_mean=config.training.reduce_mean,
            continuous=config.training.continuous,
            likelihood_weighting=config.training.likelihood_weighting,
        )
        
    def _load_pretrain_weights(self):
        # load pretrain weights
        if not self.pretrain:
            print('Train from scratch, no pretrain weights loaded')
            return
        
        print(f"loading pretrain weights from {self.pretrain_weights_path}")
        checkpoint = torch.load(self.pretrain_weights_path)
        upd_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if 'loss.w' in k:  # Skip this key
                continue
            new_key = k.replace('promptmr.', '')  # or just remove 'model.'
            upd_state_dict[new_key] = v
        self.promptmr.load_state_dict(upd_state_dict)

    def configure_optimizers(self):
        optim = losses.get_optimizer(self.config, self.model.parameters())

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.8),
            'interval': 'epoch',     # or 'step' if you want to update every batch
            'frequency': 1,          # apply every 1 epoch
            'monitor': None          # can omit for StepLR, needed for ReduceLROnPlateau
        }
        return [optim], [scheduler]
    
    def forward(self, x, t, cond=None):
        ##return self.model(x, t, cond) 
        pass  

    def training_step(self, batch, batch_idx):
        #sample = (input_img, gt, attrs, fname.name, data_slice, num_t, num_slices)
        zfilled, label, attrs, fname, data_slice, num_t, num_slices = batch

        if batch_idx < 3 and self.logger is not None and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            zfilled_grid = vutils.make_grid(zfilled[:4, None].detach().cpu(), normalize=True, scale_each=True)
            label_grid = vutils.make_grid(label[:4, None].detach().cpu(), normalize=True, scale_each=True)

            self.logger.experiment.log({
                "train_input_zfilled": wandb.Image(zfilled_grid, caption="Input (zfilled)"),
                "train_label": wandb.Image(label_grid, caption="Label"),
                "global_step": self.global_step
            })
            
        # Execute one training step
        loss, predict, target = self.train_step_fn(self.state, label, zfilled)
        #keep the first channel of predict and target
        predict = predict[:, 0:1, :, :]
        target = target[:, 0:1, :, :]
        
        #if batch_idx < 3, log predict and target
        if batch_idx < 3 and self.logger is not None and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            predict_grid = vutils.make_grid(predict[:4, None].detach().cpu(), normalize=True, scale_each=True)
            target_grid = vutils.make_grid(target[:4, None].detach().cpu(), normalize=True, scale_each=True)

            self.logger.experiment.log({
                "train_predictions": wandb.Image(predict_grid, caption="Predictions"),
                "train_targets": wandb.Image(target_grid, caption="Targets"),
                "global_step": self.global_step
            })

        self.log("train_loss", loss, prog_bar=True)

        if torch.isnan(loss):
            raise ValueError("NaN loss encountered during training.")

        #print(torch.cuda.memory_summary(device=None, abbreviated=False), flush=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        if batch_idx >= 3:
            return None
        zfilled, label, attrs, fname, data_slice, num_t, num_slices = batch
        device = zfilled.device

        # Initialize DDIM parameters
        num_steps = 100
        config = self.config
        sde = sde_lib.VPSDE(config)
        zfilled = zfilled[:, None, :, :]  # Ensure zfilled is [B, C, H, W]
        shape = zfilled.shape  # shape = [B, C, H, W]
        x = torch.randn(shape, device=device)  # Start from noise

        #concat x and zfilled
        x = torch.cat([x, zfilled], dim=1)  # shape [B, C+1, H, W]

        # If using EMA, apply it to the model
        if config.model.ema_rate > 0:
            self.state["ema"].store(self.model.parameters())
            self.state["ema"].copy_to(self.model.parameters())

        # Time discretization
        timesteps = torch.linspace(1.0, 1e-3, num_steps, device=device)

        # Reverse sampling loop (simplified DDIM style)
        for i, t in enumerate(timesteps):
            t_tensor = torch.ones((shape[0],), device=device) * t  # shape [B]
            # Predict noise or score
            score_fn = mutils.get_score_fn(
                model=self.model,
                sde=sde,
                train=False
            )
            score = score_fn(x, t_tensor)

            # Euler update
            dt = -1.0 / num_steps
            drift, diffusion = sde.sde(x, t_tensor)
            x = x + drift * dt + diffusion[:, None, None, None] * score * torch.sqrt(torch.tensor(-dt, device=score.device))

        # Compute simple MSE for validation loss
        pred = x[:, 0:1, :, :]  # Keep only the first channel for prediction
        target = label[:, None, :, :]  # Ensure target is [B, C, H, W]
        loss = F.mse_loss(pred, target)

        self.log("val_loss", loss, prog_bar=True)

        if batch_idx < 3 and self.logger is not None and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            # Keep only the first channel for logging
            recon_grid = vutils.make_grid(pred[:4, 0:1].detach().cpu(), normalize=True, scale_each=True)
            target_grid = vutils.make_grid(target[:4, 0:1].detach().cpu(), normalize=True, scale_each=True)

            self.logger.experiment.log({
                "val_reconstructions": wandb.Image(recon_grid, caption="Reconstructed"),
                "val_targets": wandb.Image(target_grid, caption="Ground Truth"),
                "global_step": self.global_step
            })
        
        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass
        
class PromptMrModule(MriModule):

    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0: int = 48,
        feature_dim: List[int] = [72,96,120],
        prompt_dim: List[int] = [24,48,72],
        sens_n_feat0: int = 24,
        sens_feature_dim: List[int] = [36,48,60],
        sens_prompt_dim: List[int] = [12,24,36],
        len_prompt: List[int] = [5,5,5],
        prompt_size: List[int] = [64,32,16],
        n_enc_cab: List[int] = [2,3,3],
        n_dec_cab: List[int] = [2,2,3],
        n_skip_cab: List[int] = [1,1,1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        learnable_prompt: bool = False,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            sens_n_feat0: Number of top-level feature channels for sense map
                estimation PromptUnet in PromptMR.
            sens_feature_dim: feature dim for each level in PromptUnet for
                sensitivity map estimation (SME) network.
            sens_prompt_dim: prompt dim for each level in PromptUnet in
                sensitivity map estimation (SME) network.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in BottleneckBlock.
            no_use_ca: not using channel attention.
            learnable_prompt: whether to set the prompt as learnable parameters.
            adaptive_input: whether to use adaptive input.
            n_buffer: number of buffer in adaptive input.
            n_history: number of historical feature aggregation, should be less than num_cascades.
            use_sens_adj: whether to use adjacent sensitivity map estimation.
            model_version: model version. Default is "promptmr_v2".
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
            compute_sens_per_coil: (bool) whether to compute sensitivity maps per coil for memory saving
            pretrain: whether to load pretrain weights
            pretrain_weights_path: path to pretrain weights
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices

        self.n_feat0 = n_feat0
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim

        self.sens_n_feat0 = sens_n_feat0
        self.sens_feature_dim = sens_feature_dim
        self.sens_prompt_dim = sens_prompt_dim

        self.len_prompt = len_prompt
        self.prompt_size = prompt_size
        self.n_enc_cab = n_enc_cab
        self.n_dec_cab = n_dec_cab
        self.n_skip_cab = n_skip_cab
        self.n_bottleneck_cab = n_bottleneck_cab

        self.no_use_ca = no_use_ca

        self.learnable_prompt = learnable_prompt
        self.adaptive_input = adaptive_input
        self.n_buffer = n_buffer
        self.n_history = n_history
        self.use_sens_adj = use_sens_adj
        # two flags for reducing memory usage
        self.use_checkpoint = use_checkpoint
        self.compute_sens_per_coil = compute_sens_per_coil
        
        self.pretrain = pretrain
        self.pretrain_weights_path = pretrain_weights_path
        
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.model_version = model_version
        PromptMR = get_model_class(f"models.{model_version}")  # Dynamically get the model class
        
        self.promptmr = PromptMR(
            num_cascades=self.num_cascades,
            num_adj_slices=self.num_adj_slices,
            n_feat0=self.n_feat0,
            feature_dim = self.feature_dim,
            prompt_dim = self.prompt_dim,
            sens_n_feat0=self.sens_n_feat0,
            sens_feature_dim = self.sens_feature_dim,
            sens_prompt_dim = self.sens_prompt_dim,
            len_prompt = self.len_prompt,
            prompt_size = self.prompt_size,
            n_enc_cab = self.n_enc_cab,
            n_dec_cab = self.n_dec_cab,
            n_skip_cab = self.n_skip_cab,
            n_bottleneck_cab = self.n_bottleneck_cab,
            no_use_ca=self.no_use_ca,
            learnable_prompt = learnable_prompt,
            n_history = self.n_history,
            n_buffer = self.n_buffer,
            adaptive_input = self.adaptive_input,
            use_sens_adj = self.use_sens_adj,
        )
        
        self._load_pretrain_weights()
        self.loss = SSIMLoss()
        
    def _load_pretrain_weights(self):
        # load pretrain weights
        if not self.pretrain:
            print('Train from scratch, no pretrain weights loaded')
            return
        
        print(f"loading pretrain weights from {self.pretrain_weights_path}")
        checkpoint = torch.load(self.pretrain_weights_path)
        upd_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if 'loss.w' in k:  # Skip this key
                continue
            new_key = k.replace('promptmr.', '')  # or just remove 'model.'
            upd_state_dict[new_key] = v
        self.promptmr.load_state_dict(upd_state_dict)

    def configure_optimizers(self):

        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # step lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]
    
    def forward(self, masked_kspace, masked_kspace_tmean, mask, num_low_frequencies, mask_type="cartesian", use_checkpoint=False, compute_sens_per_coil=False):
        return self.promptmr(masked_kspace, masked_kspace_tmean, mask, num_low_frequencies, mask_type, use_checkpoint=use_checkpoint, compute_sens_per_coil=compute_sens_per_coil)   

    def training_step(self, batch, batch_idx):
        #print batch contents 
        print(batch.mask_type)
        output_dict = self(batch.masked_kspace, batch.masked_kspace_tmean, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           use_checkpoint=self.use_checkpoint, compute_sens_per_coil=self.compute_sens_per_coil)
        output = output_dict['img_pred']
        target, output = transforms.center_crop_to_smallest(
            batch.target, output)

        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )
        self.log("train_loss", loss, prog_bar=True)

        ##! raise error if loss is nan
        if torch.isnan(loss):
            raise ValueError(f'nan loss on {batch.fname} of slice {batch.slice_num}')
        return loss


    def validation_step(self, batch, batch_idx):

        output_dict = self(batch.masked_kspace, batch.masked_kspace_tmean, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil)
        output = output_dict['img_pred']
        img_zf = output_dict['img_zf']
        target, output = transforms.center_crop_to_smallest(
            batch.target, output)
        _, img_zf = transforms.center_crop_to_smallest(
            batch.target, img_zf)
        val_loss = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            )
        cc = batch.masked_kspace.shape[1]
        centered_coil_visual = torch.log(1e-10+torch.view_as_complex(batch.masked_kspace[:,cc//2]).abs())
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "img_zf":   img_zf,
            "mask": centered_coil_visual, 
            "sens_maps": output_dict['sens_maps'][:,0].abs(),
            "output": output,
            "target": target,
            "loss": val_loss,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil)
        output = output_dict['img_pred']

        crop_size = batch.crop_size 
        crop_size = [crop_size[0][0], crop_size[1][0]] # if batch_size>1
        # detect FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        output = transforms.center_crop(output, crop_size)

        num_slc = batch.num_slc
        return {
            'output': output.cpu(), 
            'slice_num': batch.slice_num, 
            'fname': batch.fname,
            'num_slc':  num_slc
        }

