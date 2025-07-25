"""
This file contains the basic modules for the model. 
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import transforms

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, stride=stride)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, no_use_ca=False):
        super().__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        if not no_use_ca:
            self.CA = CALayer(n_feat, reduction, bias=bias)
        else:
            self.CA = nn.Identity()
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
# ---------- Prompt Block -----------------------

class PromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, learnable_prompt=False):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size), 
                                         requires_grad=learnable_prompt)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, emb):

        B, C, H, W = x.shape
        #emb is (B, 1), broadcast to (B, C)
        emb = emb.expand(-1, C)
        #emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.dec_conv3x3(prompt)

        return prompt

class PromptBlock_meta(nn.Module):
    """
    Prompt block that is driven by an external metadata tensor (B, N).
    Backward compatible:
      - If meta is None, we fallback to the old behavior (use spatial mean of x -> emb).
      - Custom _load_from_state_dict drops mismatched keys, so strict=False load won't crash.

    Args:
        prompt_dim (int): Channels of each prompt template.
        prompt_len (int): Number of prompt templates in the bank.
        prompt_size (int): Spatial size (HxW) of prompt templates before upsample.
        meta_dim (int): Dimension of the metadata vector fed in (B, N). Only used when meta is provided.
        lin_dim_legacy (int): Dimension for the legacy embedding (mean-pooled x). Keep to load old checkpoints.
        learnable_prompt (bool): Whether the prompt bank is trainable.
    """
    def __init__(self,
                 prompt_dim: int = 128,
                 prompt_len: int = 16,
                 prompt_size: int = 96,
                 meta_dim: int = 17,
                 lin_dim_legacy: int = 192,
                 learnable_prompt: bool = False):
        super().__init__()
        # Prompt bank
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size),
            requires_grad=learnable_prompt
        )
        # Two linear layers: one for meta, one kept for legacy (emb from x)
        self.linear_layer_meta = nn.Linear(meta_dim, prompt_len)
        self.linear_layer_legacy = nn.Linear(lin_dim_legacy, prompt_len)

        # Decoder conv
        self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: feature map (B, C, H, W).
            meta: optional metadata tensor (B, N). If None, fallback to legacy behavior.

        Returns:
            (B, prompt_dim, H, W) prompt feature.
        """
        B, _, H, W = x.shape

        if meta is not None:
            if meta.dim() == 1:
                meta = meta.unsqueeze(0)
            assert meta.shape[0] == B, "Batch size of meta must match x"
            prompt_weights = F.softmax(self.linear_layer_meta(meta), dim=1)  # (B, L)
        else:
            # legacy path
            emb = x.mean(dim=(-2, -1))                                     # (B, C')
            prompt_weights = F.softmax(self.linear_layer_legacy(emb), dim=1)

        # prompt_param: (1, L, D, P, P) -> (B, L, D, P, P)
        prompt_param = self.prompt_param.expand(B, -1, -1, -1, -1)
        # weight and sum
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = prompt.sum(dim=1)  # (B, D, P, P)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear", align_corners=False)
        prompt = self.dec_conv3x3(prompt)
        return prompt

    # ---- Safe checkpoint loading ----
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Drop keys with mismatched shapes (e.g., meta linear weights) to allow strict=False loading gracefully.
        """
        keys_to_drop = []
        for k, v in state_dict.items():
            if not k.startswith(prefix):
                continue
            name = k[len(prefix):]
            if name.startswith("linear_layer_meta"):
                # if shape mismatch, drop
                try:
                    tgt = dict(self.named_parameters())[name]
                    if v.shape != tgt.shape:
                        keys_to_drop.append(k)
                except KeyError:
                    keys_to_drop.append(k)
            elif name.startswith("linear_layer_legacy"):
                try:
                    tgt = dict(self.named_parameters())[name]
                    if v.shape != tgt.shape:
                        keys_to_drop.append(k)
                except KeyError:
                    keys_to_drop.append(k)

        for k in keys_to_drop:
            state_dict.pop(k)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

class PromptBlock_meta_expressive(nn.Module):
    """
    More expressive Prompt Block.
    - Outputs SAME shape as PromptBlock_meta: (B, prompt_dim, H, W)
    - Uses ONLY metadata (meta) for spatial weights.
    - Safe checkpoint loader to accept original PromptBlock / PromptBlock_meta weights.
    Args:
        in_dim (int):     Channels of input feature map x.
        prompt_dim (int): Channels per prompt template.
        prompt_len (int): Number of prompt templates.
        prompt_size (int): Template spatial size before upsample.
        meta_dim (int):   Metadata vector length (B, N).
        learnable_prompt (bool): Train prompt bank or not.
        attn_dim (int):   Hidden dim for meta MLP.
    """
    def __init__(self,
                 prompt_dim: int = 128,
                 prompt_len: int = 16,
                 prompt_size: int = 96,
                 meta_dim: int = 18,
                 learnable_prompt: bool = False,
                 attn_dim: int = 256):
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt_dim = prompt_dim
        self.prompt_size = prompt_size
        self.meta_dim = meta_dim = 18

        if meta_dim is None or meta_dim <= 0:
            raise ValueError("meta_dim must be > 0 for PromptBlock_meta_expressive")

        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size),
            requires_grad=learnable_prompt
        )
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, attn_dim),
            nn.SiLU(inplace=True),
            nn.Linear(attn_dim, prompt_len)
        )
        nn.init.xavier_uniform_(self.meta_mlp[0].weight)
        nn.init.zeros_(self.meta_mlp[0].bias)
        nn.init.xavier_uniform_(self.meta_mlp[2].weight)
        nn.init.zeros_(self.meta_mlp[2].bias)

        self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x    : (B, C, H, W)
            meta : (B, N) or None
        Returns:
            prompt_feat: (B, prompt_dim, H, W)
        """
        B, C, H, W = x.shape
        # 1) Metadata to generate spatial weights
        if meta is None or meta.numel() == 0:
            meta_weights = torch.zeros(B, self.prompt_len, device=x.device, dtype=x.dtype)
        else:
            if meta.dim() == 1:
                meta = meta.unsqueeze(0)
            assert meta.shape[0] == B, "Batch size of meta must match x"
            meta_weights = self.meta_mlp(meta)  # (B, L)

        # 2) Prepare and upsample prompt bank
        pb = self.prompt_param.expand(B, -1, -1, -1, -1)  # (B, L, D, P, P)
        pb = pb.view(B * self.prompt_len, self.prompt_dim, self.prompt_size, self.prompt_size)
        pb = F.interpolate(pb, (H, W), mode='bilinear', align_corners=False)
        pb = pb.view(B, self.prompt_len, self.prompt_dim, H, W)
        #print(f"[PromptBlock_meta_expressive] pb.shape = {pb.shape}", flush=True)

        # 3) Spatial weights from meta (broadcast across H, W)
        meta_weights = meta_weights.view(B, self.prompt_len, 1, 1, 1)
        prompt = (meta_weights * pb).sum(dim=1)  # (B, D, H, W)
        #print(f"[PromptBlock_meta_expressive] prompt.shape = {prompt.shape}", flush=True)

        # 4) Final conv
        prompt = self.dec_conv3x3(prompt)
        #print(f"[PromptBlock_meta_expressive] Final prompt.shape = {prompt.shape}", flush=True)
        return prompt

    # ---------------- Safe checkpoint loading -----------------
    '''
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Do not load any weights; keep random initialization
        keys_to_drop = [k for k in list(state_dict.keys()) if k.startswith(prefix)]
        for k in keys_to_drop:
            state_dict.pop(k, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
    '''
                                      
class DownBlock(nn.Module):
    def __init__(self, input_channel, output_channel, n_cab, kernel_size, reduction, bias, act,
                 no_use_ca=False, first_act=False):
        super().__init__()
        if first_act:
            self.encoder = [CAB(input_channel, kernel_size, reduction,bias=bias, act=nn.PReLU(), no_use_ca=no_use_ca)]
            self.encoder = nn.Sequential(
                    *(self.encoder+[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) 
                                    for _ in range(n_cab-1)]))
        else:
            self.encoder = nn.Sequential(
                *[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) 
                  for _ in range(n_cab)])
        self.down = nn.Conv2d(input_channel, output_channel,kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, n_cab, kernel_size, reduction, bias, act,
                 no_use_ca=False, n_history=0):
        super().__init__()
        # momentum layer
        self.n_history = n_history
        if n_history > 0:
            self.momentum = nn.Sequential(
                nn.Conv2d(in_dim*(n_history+1), in_dim, kernel_size=1, bias=bias),
                CAB(in_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
            )

        self.fuse = nn.Sequential(*[CAB(in_dim+prompt_dim, kernel_size, reduction,
                                        bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))

        self.ca = CAB(out_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

    def forward(self, x, prompt_dec, skip, history_feat: Optional[torch.Tensor] = None):
        # momentum layer
        if self.n_history > 0:
            if history_feat is None:
                x = torch.cat([torch.tile(x, (1, self.n_history+1, 1, 1))], dim=1)
            else:
                x = torch.cat([x, history_feat], dim=1)

            x = self.momentum(x)

        x = torch.cat([x, prompt_dec], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)

        x = self.up(x) + skip
        x = self.ca(x)

        return x


class SkipBlock(nn.Module):
    def __init__(self, enc_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super().__init__()
        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(*[CAB(enc_dim, kernel_size, reduction, bias=bias, act=act,
                                                 no_use_ca=no_use_ca) for _ in range(n_cab)])

    def forward(self, x):
        x = self.skip_attn(x)
        return x
    
    
class KspaceACSExtractor:
    '''
    Extract ACS lines from k-space data
    '''

    def __init__(self, mask_center):
        self.mask_center = mask_center
        self.low_mask_dict = {}  # avoid repeated calculation

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        get the padding size and number of low frequencies for the center mask. For fastmri and cmrxrecon dataset
        '''
        if num_low_frequencies is None or (num_low_frequencies == -1).all():
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2
        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def circular_centered_mask(self, shape, radius):
        '''
        generate a circular mask centered at the center of the image. For calgary-campinas dataset
        -shape: the shape of the mask
        -radius: the radius of the circle (ACS region)

        '''
        # radius is a tensor or int
        if type(radius) == torch.Tensor:
            # radius[0].item() # assume batch have the same radius
            radius = int(radius[0])

        center = torch.tensor(shape) // 2
        Y, X = torch.meshgrid(torch.arange(
            shape[0]), torch.arange(shape[1]), indexing='ij')
        dist_from_center = torch.sqrt(
            (X - center[1]) ** 2 + (Y - center[0]) ** 2)
        mask = (dist_from_center <= radius).float()
        return mask.unsqueeze(0).unsqueeze(-1)

    def __call__(self, masked_kspace: torch.Tensor,
                mask: torch.Tensor,
                num_low_frequencies: Optional[int] = None,
                mask_type: Tuple[str] = ("cartesian",),
                ) -> torch.Tensor:
        if self.mask_center:
            mask_type = mask_type[0]  # assume the same type in a batch
            mask_type = 'cartesian' if mask_type in ['uniform', 'kt_uniform', 'kt_random'] else mask_type
            print(f"Shape of mask: {mask.shape}")
                # reshape mask to [b, h, w, two] if needed
            if mask.ndim == 5:
                #adj_sli = mask.shape[1] 
                center_idx = mask.shape[1] // 2
                mask = mask[:, center_idx]  # shape becomes [b, h, w, 2]
            elif mask.ndim != 4:
                raise ValueError(f"Expected mask to have 4 or 5 dimensions, but got shape {mask.shape}")

            if mask_type == 'kt_radial':
                mask_low = torch.zeros_like(mask)
                b, h, w, two = mask.shape
                h_left = h // 2 - num_low_frequencies // 2
                w_left = w // 2 - num_low_frequencies // 2
                h_left = int(h_left)  # Ensure h_left is an integer
                w_left = int(w_left)  # Ensure w_left is an integer
                num_low_frequencies = int(num_low_frequencies)  # Ensure num_low_frequencies is an integer

                mask_low[:, h_left:h_left + num_low_frequencies, w_left:w_left + num_low_frequencies, :] = \
                    mask[:, h_left:h_left + num_low_frequencies, w_left:w_left + num_low_frequencies, :]
                masked_kspace_acs = masked_kspace * mask_low

            elif mask_type == 'cartesian':
                # unsqueeze to match expected shape for get_pad_and_num_low_freqs (which expects 5D)
                pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask.unsqueeze(1), num_low_frequencies)
                # batched_mask_center expects 5D input
                masked_kspace_acs = transforms.batched_mask_center(masked_kspace.unsqueeze(1), pad, pad + num_low_freqs)
                masked_kspace_acs = masked_kspace_acs.squeeze(1)  # back to 4D

            elif mask_type == 'poisson_disc':
                ss = masked_kspace.shape[-3:-1]  # (h,w)
                if ss not in self.low_mask_dict:
                    mask_low = self.circular_centered_mask(masked_kspace.shape[-3:-1], num_low_frequencies)  # (1, h, w, 1)
                    mask_low = mask_low.to(masked_kspace.device)  # [1, h, w, 1]
                    self.low_mask_dict[ss] = mask_low
                else:
                    mask_low = self.low_mask_dict[ss]
                masked_kspace_acs = masked_kspace * mask_low

            else:
                raise ValueError('mask_type should be cartesian or poisson_disc')

            # Expand to match mask's depth dimension (e.g., number of slices)
            if masked_kspace_acs.ndim == 4:
                # [b, c, h, w, 2] → [b, c * t, h, w, 2]
                masked_kspace_acs = masked_kspace_acs.unsqueeze(1)
                
            return masked_kspace_acs

        else:
            return masked_kspace