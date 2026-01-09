#!/usr/bin/env python3
"""
Batch pediatric-FOV simulation:
  - Convert k-space -> coil images
  - Shrink content (H,W) with anti-aliased interpolation
  - Apply 2D Tukey edge taper to avoid seams
  - Embed into zero canvas (same matrix size)
  - FFT back to k-space, recompute RSS
  - Save H5 (same schema & shapes) + before/after PNGs (same window)

Assumes input H5 has datasets: "kspace", "reconstruction_rss".
"""

import os, sys, h5py, numpy as np, torch
import imageio.v3 as iio
import torch.nn.functional as F

# ========= Repo deps =========
REPO_ROOT = "/common/lidxxlab/cmrchallenge/code/PromptMR-plus"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from data.transforms import to_tensor
from mri_utils import ifft2c, fft2c, rss_complex

# ========= I/O config =========
INPUT_BASE   = "/common/lidxxlab/cmrchallenge/data/CMR2025_chushu/CMR_2025_3T_only/train"
OUTPUT_BASE  = "/common/lidxxlab/cmrchallenge/data/CMR2025_chushu/CMR2025_pad_FOV_3T_only/train"
PNG_OUT_BASE = "/common/lidxxlab/cmrchallenge/data/CMR2025_chushu/CMR2025_pedsFOV_png_bicubic"
SUFFIX       = "_pedsFOV"
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PNG_OUT_BASE, exist_ok=True)

# ========= Resume/skip behavior =========
SKIP_IF_EXISTS = True

# ========= Shrink/taper params =========
SCALE_H = 0.70          # 0<scale<=1  (smaller => heart appears smaller)
SCALE_W = 0.70
TUKEY_A = 0.25          # edge taper inside the shrunk region (0: none, 1: Hann-like)

# ========= PNG config =========
PNG_PERCENTILES = (1.0, 99.0)   # window from BEFORE image
USE_SAME_WINDOW = True          # render AFTER with BEFORE window

# ---------------- helpers ----------------
def mirror_path(src_path, input_base, output_base, suffix=""):
    rel = os.path.relpath(src_path, start=input_base)
    dst = os.path.join(output_base, rel)
    if suffix:
        root, ext = os.path.splitext(dst)
        dst = f"{root}{suffix}{ext}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return dst

def mirror_png_dir_for_dst(dst_h5_path, output_base, png_out_base):
    rel_dir  = os.path.relpath(os.path.dirname(dst_h5_path), start=output_base)
    stem     = os.path.splitext(os.path.basename(dst_h5_path))[0]
    png_dir  = os.path.join(png_out_base, rel_dir, f"{stem}_png")
    os.makedirs(png_dir, exist_ok=True)
    return png_dir

def ensure_ri_np(k_np: np.ndarray) -> np.ndarray:
    """Return RI format (...,H,W,2) float32 from complex or RI input."""
    if np.iscomplexobj(k_np):
        return np.stack([k_np.real, k_np.imag], axis=-1).astype(np.float32, copy=False)
    assert k_np.shape[-1] == 2, f"Expected RI last dim=2; got {k_np.shape}"
    return k_np.astype(np.float32, copy=False)

def tukey2d(h, w, alpha=0.25, device="cpu", dtype=torch.float32):
    """2D Tukey window (outer product of 1D Tukeys), size h×w."""
    if alpha <= 0:
        wy = torch.ones(h, device=device, dtype=dtype)
        wx = torch.ones(w, device=device, dtype=dtype)
    else:
        def tukey1d(n, a):
            t = torch.linspace(0, 1, n, device=device, dtype=dtype)
            win = torch.ones_like(t)
            L = a / 2.0
            if a > 0:
                idx = t < L
                win[idx] = 0.5 * (1 + torch.cos(torch.pi * (2*t[idx]/a - 1)))
                idx = t > (1 - L)
                win[idx] = 0.5 * (1 + torch.cos(torch.pi * (2*(1 - t[idx])/a - 1)))
            return win
        wy = tukey1d(h, alpha)
        wx = tukey1d(w, alpha)
    return wy[:, None] * wx[None, :]

def _window_from(arr, q=(1.0, 99.0)):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, q)
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)

def _to_uint8(img, lo, hi):
    x = np.clip((img - lo) / (hi - hi if hi == lo else hi - lo), 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def save_pair_pngs(rss_before: np.ndarray, rss_after: np.ndarray, out_dir: str, base_prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    def save_2d_pair(img_b, img_a, stem):
        lo, hi = _window_from(img_b, q=PNG_PERCENTILES)
        iio.imwrite(os.path.join(out_dir, f"{stem}_before.png"), _to_uint8(img_b, lo, hi))
        iio.imwrite(os.path.join(out_dir, f"{stem}_after.png"),  _to_uint8(img_a, lo, hi))
    if rss_before.ndim == 2:
        save_2d_pair(rss_before, rss_after, f"{base_prefix}")
    elif rss_before.ndim == 3:
        for n in range(rss_before.shape[0]):
            save_2d_pair(rss_before[n], rss_after[n], f"{base_prefix}_s{n:03d}")
    elif rss_before.ndim == 4:
        T, Z = rss_before.shape[:2]
        for t in range(T):
            for z in range(Z):
                save_2d_pair(rss_before[t, z], rss_after[t, z], f"{base_prefix}_t{t:03d}_z{z:03d}")
    else:
        print(f"[WARN] Unsupported RSS ndim={rss_before.ndim}; skipping PNG save.")

# -------- shrink + taper + embed (same size) --------
def shrink_with_taper(img_ri: torch.Tensor, scale_h: float, scale_w: float, tukey_alpha: float):
    """
    img_ri: (Nt,Nz,Nc,H,W,2) complex in RI.
    Returns: (Nt,Nz,Nc,H,W,2) with content shrunk + edge-tapered, centered on zero canvas.
    """
    assert img_ri.ndim == 6 and img_ri.shape[-1] == 2
    Nt, Nz, Nc, H, W, _ = img_ri.shape
    Hs = max(2, int(round(H * scale_h)))
    Ws = max(2, int(round(W * scale_w)))
    h0 = (H - Hs) // 2
    w0 = (W - Ws) // 2

    B = Nt * Nz * Nc
    x = img_ri.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # (B,2,H,W)

    # Anti-aliased bilinear downscale
    #x_small = F.interpolate(x, size=(Hs, Ws), mode="bilinear", align_corners=False, antialias=True)
    x_small = F.interpolate(x, size=(Hs, Ws), mode="bicubic", align_corners=False, antialias=True)

    # Tukey edge taper to avoid rectangular seam
    win2d = tukey2d(Hs, Ws, alpha=tukey_alpha, device=x_small.device, dtype=x_small.dtype)  # (Hs,Ws)
    x_small = x_small * win2d[None, None, ...]  # apply to both real & imag

    # Place on zero canvas of original size
    canvas = torch.zeros(B, 2, H, W, dtype=x.dtype, device=x.device)
    canvas[..., h0:h0+Hs, w0:w0+Ws] = x_small

    return canvas.permute(0, 2, 3, 1).reshape(Nt, Nz, Nc, H, W, 2)

# ---------------- core per-file pipeline ----------------
def process_one(src_path, dst_path, png_dir=None):
    # --- read ---
    with h5py.File(src_path, "r") as f:
        k_orig   = f["kspace"][()]
        rss_orig = f["reconstruction_rss"][()]
        attrs    = dict(f.attrs)

    was_complex    = np.iscomplexobj(k_orig)
    kshape_orig    = k_orig.shape
    kdtype_orig    = k_orig.dtype
    rss_shape_orig = rss_orig.shape

    # --- to RI torch ---
    k_ri   = to_tensor(ensure_ri_np(k_orig))      # (...,H,W,2)
    img_ri = ifft2c(k_ri)                         # (Nt,Nz,Nc,H,W,2)

    # --- shrink + taper + embed (same H,W) ---
    img_new = shrink_with_taper(img_ri, SCALE_H, SCALE_W, TUKEY_A)

    # --- back to k-space & RSS ---
    k_new_ri = fft2c(img_new)
    rss_new  = rss_complex(img_new, dim=-3).cpu().numpy()

    # --- convert k back to input format ---
    k_ri_np_new = k_new_ri.detach().cpu().numpy()
    if was_complex:
        k_new = (k_ri_np_new[..., 0] + 1j * k_ri_np_new[..., 1]).astype(kdtype_orig, copy=False)
    else:
        k_new = k_ri_np_new.astype(kdtype_orig, copy=False)

    # --- verify shapes unchanged ---
    if k_new.shape != kshape_orig:
        raise AssertionError(f"kspace shape changed: new {k_new.shape} vs orig {kshape_orig}")
    if rss_new.shape != rss_shape_orig:
        if rss_new.size == np.prod(rss_shape_orig):
            rss_new = rss_new.reshape(rss_shape_orig)
        else:
            raise AssertionError(f"reconstruction_rss shape changed: new {rss_new.shape} vs orig {rss_shape_orig}")

    # --- PNGs (same-window) ---
    if png_dir is None:
        png_dir = mirror_png_dir_for_dst(dst_path, OUTPUT_BASE, PNG_OUT_BASE)
    src_stem = os.path.splitext(os.path.basename(src_path))[0]
    save_pair_pngs(rss_orig, rss_new, png_dir, src_stem)

    # --- write H5 (same schema) ---
    H_attr = k_new.shape[-2] if was_complex else k_new.shape[-3]
    W_attr = k_new.shape[-1] if was_complex else k_new.shape[-2]
    attrs_out = dict(attrs)
    attrs_out["max"]   = float(rss_new.max()) if np.isfinite(rss_new).any() else 0.0
    attrs_out["norm"]  = float(np.linalg.norm(rss_new)) if np.isfinite(rss_new).any() else 0.0
    attrs_out["shape"] = tuple(k_new.shape)
    attrs_out["padding_left"]  = 0
    attrs_out["padding_right"] = int(W_attr)
    attrs_out["encoding_size"] = (int(H_attr), int(W_attr), 1)
    attrs_out["recon_size"]    = (int(H_attr), int(W_attr), 1)
    attrs_out.setdefault("acquisition", attrs_out.get("acquisition", "unknown"))
    attrs_out.setdefault("patient_id",  attrs_out.get("patient_id", os.path.splitext(os.path.basename(dst_path))[0]))

    with h5py.File(dst_path, "w") as f:
        f.create_dataset("kspace", data=k_new)
        f.create_dataset("reconstruction_rss", data=rss_new.astype(np.float32, copy=False))
        for k, v in attrs_out.items():
            f.attrs[k] = v

    print(f"[OK] {os.path.basename(dst_path)} | kspace {k_new.shape} | rss {rss_new.shape} (unchanged)")

# ---------------- driver ----------------
def process_tree(input_base=INPUT_BASE, output_base=OUTPUT_BASE, suffix=SUFFIX):
    print(f"Input base : {input_base}")
    print(f"Output base: {output_base}")
    print(f"PNG base   : {PNG_OUT_BASE}")
    n_done, n_skip, n_fail = 0, 0, 0
    for dp, _, files in os.walk(input_base):
        for fn in sorted(files):
            if not fn.lower().endswith(".h5"):
                continue
            src = os.path.join(dp, fn)
            dst = mirror_path(src, input_base, output_base, suffix)

            if SKIP_IF_EXISTS and os.path.exists(dst):
                print(f"[SKIP] exists: {dst}")
                n_skip += 1
                continue

            png_dir = mirror_png_dir_for_dst(dst, output_base, PNG_OUT_BASE)

            try:
                process_one(src, dst, png_dir=png_dir)
                n_done += 1
            except Exception as e:
                print(f"[FAIL] {src}: {e}")
                n_fail += 1
    print(f"Done. Wrote {n_done} files. Skipped {n_skip}. Failed {n_fail}.")

if __name__ == "__main__":
    process_tree()