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
from skimage.metrics import structural_similarity as sk_ssim, peak_signal_noise_ratio as sk_psnr
from mri_utils.sampling_utils import CG, clear, get_mask, nchw_comp_to_real, real_to_nchw_comp, normalize_np, get_beta_schedule
from mri_utils.mri import MulticoilMRI
import torchvision.transforms as T
from torchvision.utils import save_image

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

class DDS_Sampling(MriModule):

    def __init__(self):
        super().__init__()
        self.config = config = get_config()
        self.save_hyperparameters()
        self.pretrain = config.model.pretrain
        self.pretrain_weights_path = config.model.pretrain_weights_path

        self.model = mutils.create_model(config)
        #wandb.watch(self.model, log='all', log_freq=100)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=config.model.ema_rate)
        self._load_pretrain_weights()
        self.scaler = get_data_scaler(config)
        self.optimizer = losses.get_optimizer(config, self.model.parameters())
        self.state = dict(optimizer=self.optimizer, model=self.model, ema=self.ema, step=0)

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

    def setup(self, stage: str):
        if self.logger is not None and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            wandb.watch(self.model, log='all', log_freq=100)
        
    def _load_pretrain_weights(self):
        # load pretrain weights
        if not self.pretrain:
            print('Train from scratch, no pretrain weights loaded')
            return
        
        print(f"loading pretrain weights from {self.pretrain_weights_path}")
        checkpoint = torch.load(self.pretrain_weights_path)
        upd_state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if it exists
        upd_state_dict = {k.replace('model.', ''): v for k, v in upd_state_dict.items() if 'model.' in k}
        self.model.load_state_dict(upd_state_dict, strict=False)

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
        pass
    
    def validation_step(self, batch, batch_idx):
        # Expected batch tuple:
        #   y_kspace_ri : (B, Nc, H, W, 2)  real/imag split along channel dim
        #   sens_maps_ri: (B, Nc, H, W, 2)
        #   mask        : (B, 1,   H, W)
        #   zf_img      : (B, 1,   H, W)   zero‑filled recon (real)
        #   target_img  : (B, 1,   H, W)   ground‑truth magnitude / real
        # sample = (input_img, target, masked_kspace, sens, mask, target, attrs, fname.name, data_slice, num_t, num_slices)
        zf_img, target, metadata, y_kspace_ri, sens_maps_ri, mask, attrs, _, _, _, _ = batch
        y_kspace_ri = y_kspace_ri.squeeze(1)
        sens_maps_ri = sens_maps_ri.squeeze(1)

        # save sens_map as Nc*2 pngs
        
        if batch_idx < 3:
            save_dir = "/common/lidxxlab/Yifan/PromptMR-plus/experiments/debug/sens_map"
            os.makedirs(save_dir, exist_ok=True)
            for i in range(mask.shape[1]):
                sens_map = sens_maps_ri[0, i].cpu().numpy()
                sens_map_real = sens_map[..., 0]
                #convert to 0-1 range tensors
                sens_map_real = (sens_map_real - sens_map_real.min()) / (sens_map_real.max() - sens_map_real.min())
                #save as png
                sens_map_real = torch.tensor(sens_map_real, dtype=torch.float32)
                sens_map_real = sens_map_real.unsqueeze(0).unsqueeze(0)
                save_image(sens_map_real, os.path.join(save_dir, f"sens_map_{i}_real.png"))
        

        def smart_squeeze(x):
            shape = x.shape
            # Squeeze dimensions 2, 3, and -1 if they're singleton
            for dim in [1, 2, -1]:
                if x.size(dim) == 1:
                    x = x.squeeze(dim)
            return x
        
        mask = smart_squeeze(mask)  # Ensure mask is squeezed correctly
        #mask is (B, 1, 1, 1, W, 1), squeeze to (B, 1, W)
        device = y_kspace_ri.device
        #print size of kspace, sens map and mask for debugging
        print('y_kspace_ri shape:', y_kspace_ri.shape, 'sens_maps_ri shape:', sens_maps_ri.shape, 'mask shape:', mask.shape, 'zf_img shape:', zf_img.shape, 'target shape:', target.shape, flush=True)
        
        def ri2complex(t):
            # t: (B, C, H, W, 2) -> complex tensor (B, C, H, W)
            return torch.complex(t[..., 0], t[..., 1])

        B, Nc, H, W, _ = y_kspace_ri.shape
        y_kspace = ri2complex(y_kspace_ri).to(device)
        sens_maps = ri2complex(sens_maps_ri).to(device)
        zf_img = zf_img.to(device)        # conditioning image
        target = target.to(device)
        pad_top = attrs["pad_top"]
        pad_bottom = attrs["pad_bottom"]
        pad_left = attrs["pad_left"]
        pad_right = attrs["pad_right"]

        input_mean = attrs["input_mean"]
        input_std = attrs["input_std"]
        
        A_funcs = MulticoilMRI(image_size=H, mask=mask.to(device), sens=sens_maps)
        A  = lambda z: A_funcs.A(z)
        AT = lambda z: A_funcs.AT(z)
        Ap = lambda z: A_funcs.A_dagger(z)

        def Acg_noise(x, gamma=0.01):
            return x + gamma * AT(A(x))
            
        def Acg(x):
            return AT(A(x))

        config = self.config
        sde = sde_lib.VPSDE(config)
        bcg = AT(y_kspace)
        bcg_abs = bcg.abs()
        print(f"bcg: min={bcg_abs.min().item():.6f}, max={bcg_abs.max().item():.6f}, mean={bcg_abs.mean().item():.6f}", flush=True)

        # ---- Debug: check if A(target) ≈ y_kspace (b) ----
        '''
        with torch.no_grad():
            pad_top    = attrs["pad_top"]
            pad_bottom = attrs["pad_bottom"]
            pad_left   = attrs["pad_left"]
            pad_right  = attrs["pad_right"]
            target_mean = attrs["target_mean"]
            target_std = attrs["target_std"]

            # target was (B, H_pad, W_pad)
            tgt_unpad = target[..., pad_top:target.shape[-2]-pad_bottom,
                                        pad_left:target.shape[-1]-pad_right]  # (B, H0, W0)
            
            #de-normalize target
            tgt_unpad = tgt_unpad * target_std + target_mean

            #print min, max, mean of tgt_unpad
            print(f"tgt_unpad: min={tgt_unpad.min().item():.6f}, max={tgt_unpad.max().item():.6f}, mean={tgt_unpad.mean().item():.6f}", flush=True)
        '''

        num_steps = 1000
        timesteps = torch.linspace(1.0, 1e-3, num_steps, device=device)

        # Initialize x with noise and conditioning zero-filled image
        x = torch.randn_like(target)
        #x from [B, H, W] to [B, 1, H, W] for compatibility
        x = x.unsqueeze(1)  # shape (B, 1, H, W)
        zf_img = zf_img.unsqueeze(1)
        target = target.unsqueeze(1)  # shape (B, 1, H, W)
        # concatenate conditioning zero‑filled image as second channel
        x = torch.cat([x, zf_img], dim=1)   # shape (B,2,H,W)
        # print the shape of x
        print("x shape:", x.shape, flush=True)

        model = self.state["model"]
        if config.model.ema_rate > 0:
            self.state["ema"].store(self.model.parameters())
            self.state["ema"].copy_to(self.model.parameters())

        for j in range(num_steps):
            t = timesteps[j]
            t_tensor = torch.ones((B,), device=device) * t
            t_tensor = t_tensor.to(dtype=torch.float32)

            at = sde.alphas_cumprod[j]
            at_next = sde.alphas_cumprod[j+1] if j < num_steps - 1 else torch.tensor(0., device=device)
            eta = 0.1
            c1 = (1 - at_next).sqrt() * eta
            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
            #c1 = ((1 - at_next) / (1 - at)).sqrt() * (1 - at / at_next).sqrt()
            #c2 = ((1 - at_next) - c1**2 * (1 - at))**0.5

            with torch.no_grad():
                img = x[:, :1]  
                '''
                #save img as a png to save_dir
                save_dir = "/common/lidxxlab/Yifan/PromptMR-plus/experiments/debug"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"val_img_step{batch_idx}_t{j}.png")
                img_to_save = img.clone()
                if torch.is_complex(img_to_save):
                    img_to_save = img_to_save.abs()
                img_to_save = img_to_save[:1]  # take the first sample in batch
                img_to_save = T.Normalize(0, 1)(img_to_save) if img_to_save.max() > 1 else img_to_save
                save_image(img_to_save, save_path)
                '''
                x = x.to(dtype=torch.float32)
                # --- v-prediction model output ---
                v_pred = model(x, t_tensor, metadata)[:, :1]

                # Precompute sqrt(alpha_t) and sqrt(1-alpha_t)
                sqrt_at   = at.sqrt()
                sqrt_oma  = (1. - at).sqrt()

                # Recover epsilon and x0 from v formulation (see Imagen/DDPM-v)
                eps_pred = sqrt_at * v_pred + sqrt_oma * img            # ε_t
                x0_t     = sqrt_at * img - sqrt_oma * v_pred            # x0 estimate

                '''
                #save x0_t as a png
                x0_t = x0_t.squeeze(1)  # remove channel dimension
                x0_t = x0_t.abs()
                x0_t = T.Normalize(0, 1)(x0_t) if x0_t.max() > 1 else x0_t
                save_path = os.path.join(save_dir, f"x0_t_step{batch_idx}_t{j}.png")
                save_image(x0_t, save_path)
                '''

                x0_t = x0_t * input_std + input_mean

                x0_t = x0_t[..., pad_top:-pad_bottom, pad_left:-pad_right]

                x0_t = torch.complex(x0_t, torch.zeros_like(x0_t))
                x0_hat = CG(Acg_noise, bcg, x0_t, n_inner=10, eps=1e-8)

                x0_hat_real = x0_hat.abs()  # Or use `.abs()` if magnitude is desired

                if j != num_steps - 1:
                    input_mean = x0_hat_real.mean()
                    input_std = x0_hat_real.std()
                    x0_hat_real = (x0_hat_real - input_mean) / input_std
                    x0_hat_real = F.pad(x0_hat_real, (pad_left, pad_right, pad_top, pad_bottom))
                    img_next = at_next.sqrt() * x0_hat_real + c1 * torch.randn_like(x0_hat_real) + c2 * eps_pred
                    x = torch.cat([img_next, zf_img], dim=1)
                else:
                    img_next = x0_hat_real
                '''
                #print c1, c2, v_pred, eps_pred
                print(f"Step {j}: at={at.item():.6f}, at_next={at_next.item():.6f}, c1={c1.item():.6f}, c2={c2.item():.6f}", flush=True)
                print(f"v_pred[{j}]: min={v_pred.min().item():.6f}, max={v_pred.max().item():.6f}, mean={v_pred.mean().item():.6f}", flush=True)
                print(f"eps_pred[{j}]: min={eps_pred.min().item():.6f}, max={eps_pred.max().item():.6f}, mean={eps_pred.mean().item():.6f}", flush=True)
                print(f"x0_hat_real[{j}]: min={x0_hat_real.min().item():.6f}, max={x0_hat_real.max().item():.6f}, mean={x0_hat_real.mean().item():.6f}", flush=True)
                print(f"img_next[{j}]: min={img_next.min().item():.6f}, max={img_next.max().item():.6f}, mean={img_next.mean().item():.6f}", flush=True)
                print(f"img[{j}]: min={img.min().item():.6f}, max={img.max().item():.6f}, mean={img.mean().item():.6f}", flush=True)
                '''
                

        recon = img_next.abs() if torch.is_complex(recon:=img_next) else img_next
        target_mean = attrs["target_mean"]
        target_std = attrs["target_std"]
        #unpad target
        target = target[..., pad_top:target.shape[-2]-pad_bottom,
                        pad_left:target.shape[-1]-pad_right]
        target = target * target_std + target_mean
        gt    = target
        #print shape of gt and recon
        print(f"gt shape: {gt.shape}, recon shape: {recon.shape}", flush=True)
        mse   = F.mse_loss(recon, gt).item()
        nmse  = mse / (gt.pow(2).mean().item() + 1e-8)
        psnr  = sk_psnr(gt.cpu().numpy()[0,0], recon.cpu().numpy()[0,0], data_range=gt.max().item() - gt.min().item())
        ssim  = sk_ssim(gt.cpu().numpy()[0,0], recon.cpu().numpy()[0,0], data_range=gt.max().item() - gt.min().item())
        #print to output log
        print(f"Validation Step {batch_idx}: MSE={mse:.4f}, PSNR={psnr:.4f}, SSIM={ssim:.4f}, NMSE={nmse:.4f}", flush=True)

        if self.logger is not None and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            recon_grid = vutils.make_grid(recon[:4].detach().cpu(), normalize=True, scale_each=True)
            target_grid = vutils.make_grid(gt[:4].detach().cpu(), normalize=True, scale_each=True)
            self.logger.experiment.log({
                "val_reconstructions": wandb.Image(recon_grid, caption="Reconstructed"),
                "val_targets": wandb.Image(target_grid, caption="Ground Truth"),
                "global_step": self.global_step
            })
        
        self.log("psnr", psnr, prog_bar=True)
        self.log("ssim", ssim, prog_bar=True)
        self.log("nmse", nmse, prog_bar=True)
        return {"val_loss": mse}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass