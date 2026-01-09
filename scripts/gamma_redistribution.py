#!/usr/bin/env python3
import os, sys, h5py, numpy as np, torch
import imageio.v3 as iio

# ========= Repo deps =========
REPO_ROOT = "/common/lidxxlab/cmrchallenge/code/PromptMR-plus"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from data.transforms import to_tensor
from mri_utils import ifft2c, fft2c, rss_complex

# ========= I/O config =========
INPUT_BASE   = "/common/lidxxlab/cmrchallenge/data/CMR2025/Processed/train"
OUTPUT_BASE  = "/common/lidxxlab/cmrchallenge/data/CMR2025_chushu/CMR2025_contrast_enhanced_new_4D1025/train"
PNG_OUT_BASE = "/common/lidxxlab/cmrchallenge/data/CMR2025_chushu/CMR2025_contrast_enhanced_new_4D1025_png/train"
SUFFIX       = "_contrast_enhanced"
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PNG_OUT_BASE, exist_ok=True)

# ========= Resume/skip behavior =========
SKIP_IF_AUGMENTED = True  # skip if augmented .h5 already exists

# ========= Gamma augmentation config =========
GAMMA_MIN   = 0.9
GAMMA_MAX   = 1.3
PRINT_GAMMA = True

# ========= PNG saving config =========
SAVE_PNGS        = True
PNG_PERCENTILES  = (1.0, 99.0)
USE_SAME_WINDOW  = True

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

# ----------------- Random gamma augmentation (ND) -----------------
@torch.no_grad()
def gamma_augmentation_magND_torch(magND: torch.Tensor,
                                   gamma_min: float = 0.5,
                                   gamma_max: float = 1.5,
                                   print_gamma: bool = False,
                                   fixed_gamma: float | None = None) -> tuple[torch.Tensor, float]:
    """
    Apply a single gamma to an arbitrary-shaped magnitude tensor.
    Returns (mag_out, gamma_used).
    """
    mag = magND.clamp_min(0).to(torch.float64)
    vmax = float(mag.max())
    if vmax <= 0.0:
        g = fixed_gamma if fixed_gamma is not None else gamma_min
        return torch.zeros_like(magND, dtype=torch.float32), g
    mag_norm = mag / vmax
    g = fixed_gamma if fixed_gamma is not None else \
        (torch.rand(1, device=mag.device) * (gamma_max - gamma_min) + gamma_min).item()
    if print_gamma:
        print(f"gamma: {g:.4f}")
    mag_norm_enh = mag_norm ** g
    mag_out = (mag_norm_enh * vmax).to(magND.dtype)
    return mag_out, g

@torch.no_grad()
def apply_gamma_per_slice_4dvolume(img_ri: torch.Tensor,
                                   gamma_min: float = 0.5,
                                   gamma_max: float = 1.5,
                                   print_gamma: bool = False) -> torch.Tensor:
    """
    Apply gamma on magnitude per z-slice, uniformly to the full [Nx,Ny,Ncoil,Nt] block.
    Input img_ri: (Nt, Nz, Nc, H, W, 2)  -> returns same shape.
    One gamma per z is sampled and applied across all coils and time frames.
    """
    assert img_ri.ndim == 6 and img_ri.shape[-1] == 2, f"Unexpected img_ri shape {img_ri.shape}"
    Nt, Nz, Nc, H, W, _ = img_ri.shape
    out = img_ri.clone()

    real = img_ri[..., 0]
    imag = img_ri[..., 1]
    mag  = torch.sqrt(real**2 + imag**2)
    phs  = torch.atan2(imag, real)

    for z in range(Nz):
        # Block shape for this z: (Nt, Nc, H, W) == [Nt, Ncoil, Nx, Ny]
        mag_block = mag[:, z, ...]
        mag_enh_block, g = gamma_augmentation_magND_torch(
            mag_block, gamma_min=gamma_min, gamma_max=gamma_max, print_gamma=False
        )
        if print_gamma:
            print(f"[z={z:03d}] gamma={g:.4f}")

        out[:, z, ..., 0] = mag_enh_block * torch.cos(phs[:, z, ...])
        out[:, z, ..., 1] = mag_enh_block * torch.sin(phs[:, z, ...])

    return out

# ----------------- PNG helpers -----------------
def _to_uint8(img, lo, hi):
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(img, dtype=np.uint8)
    x = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def _save_pair_pngs(rss_before: np.ndarray,
                    rss_after:  np.ndarray,
                    out_dir: str,
                    base_prefix: str,
                    q=(1.0, 99.0),
                    use_same_window=True):
    os.makedirs(out_dir, exist_ok=True)

    def window_from(arr):
        arr_finite = arr[np.isfinite(arr)]
        if arr_finite.size == 0:
            return 0.0, 1.0
        lo, hi = np.percentile(arr_finite, q)
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    def save_2d_pair(img_b, img_a, stem):
        lo, hi = window_from(img_b) if use_same_window else window_from(img_b)
        iio.imwrite(os.path.join(out_dir, f"{stem}_before.png"), _to_uint8(img_b, lo, hi))
        iio.imwrite(os.path.join(out_dir, f"{stem}_after.png"),  _to_uint8(img_a, lo, hi))

    if rss_before.ndim == 2:
        save_2d_pair(rss_before, rss_after, f"{base_prefix}")
    elif rss_before.ndim == 3:
        N = rss_before.shape[0]
        for n in range(N):
            save_2d_pair(rss_before[n], rss_after[n], f"{base_prefix}_t{n:03d}")
    elif rss_before.ndim == 4:
        T, Z = rss_before.shape[:2]
        for t in range(T):
            for z in range(Z):
                save_2d_pair(rss_before[t, z], rss_after[t, z], f"{base_prefix}_t{t:03d}_z{z:03d}")
    else:
        print(f"[WARN] Unsupported RSS ndim={rss_before.ndim}; skipping PNG save.")

# ----------------- pipeline -----------------
def process_one(src_path, dst_path, png_dir=None,
                gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX, print_gamma=PRINT_GAMMA):
    # --- read originals ---
    with h5py.File(src_path, "r") as f:
        k_orig   = f["kspace"][()]
        rss_orig = f["reconstruction_rss"][()]
        attrs    = dict(f.attrs)  # kept if you need later

    was_complex    = np.iscomplexobj(k_orig)
    kshape_orig    = k_orig.shape
    kdtype_orig    = k_orig.dtype
    rss_shape_orig = rss_orig.shape

    # --- convert to RI ---
    if was_complex:
        k_ri_np = np.stack([k_orig.real, k_orig.imag], axis=-1).astype(np.float32, copy=False)
    else:
        assert k_orig.shape[-1] == 2, f"Unexpected RI layout: {k_orig.shape}"
        k_ri_np = k_orig.astype(np.float32, copy=False)

    with torch.no_grad():
        k_ri   = to_tensor(k_ri_np)   # (...,H,W,2)
        img_ri = ifft2c(k_ri)         # (Nt, Nz, Nc, H, W, 2)

        # === Gamma augmentation per z on full [Nx,Ny,Ncoil,Nt] ===
        img_ri_enh = apply_gamma_per_slice_4dvolume(
            img_ri, gamma_min=gamma_min, gamma_max=gamma_max, print_gamma=print_gamma
        )

        # back to k-space / RSS
        k_ri_enh = fft2c(img_ri_enh)
        rss_new  = rss_complex(img_ri_enh, dim=-3).cpu().numpy()

        # convert back to original on-disk format
        k_ri_np_new = k_ri_enh.detach().cpu().numpy()
        if was_complex:
            k_new = (k_ri_np_new[..., 0] + 1j * k_ri_np_new[..., 1]).astype(kdtype_orig, copy=False)
        else:
            k_new = k_ri_np_new.astype(kdtype_orig, copy=False)

    # --- VERIFY shapes ---
    if k_new.shape != kshape_orig:
        raise AssertionError(f"kspace shape changed: new {k_new.shape} vs orig {kshape_orig}")
    if rss_new.shape != rss_shape_orig:
        if rss_new.size == np.prod(rss_shape_orig):
            rss_new = rss_new.reshape(rss_shape_orig)
        else:
            raise AssertionError(f"reconstruction_rss shape changed: new {rss_new.shape} vs orig {rss_shape_orig}")

    # --- PNGs (separate base) ---
    if SAVE_PNGS:
        if png_dir is None:
            png_dir = mirror_png_dir_for_dst(dst_path, OUTPUT_BASE, PNG_OUT_BASE)
        src_stem = os.path.splitext(os.path.basename(src_path))[0]
        _save_pair_pngs(
            rss_before=rss_orig,
            rss_after=rss_new,
            out_dir=png_dir,
            base_prefix=src_stem,
            q=PNG_PERCENTILES,
            use_same_window=USE_SAME_WINDOW
        )

    # --- write result ---
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

    print(f"[OK] saved {os.path.basename(dst_path)} | kspace {k_new.shape} == {kshape_orig} | rss {rss_new.shape} == {rss_shape_orig}")

def process_tree(input_base=INPUT_BASE, output_base=OUTPUT_BASE, suffix=SUFFIX,
                 gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX, print_gamma=PRINT_GAMMA):
    print(f"Input base : {input_base}")
    print(f"Output base: {output_base}")
    print(f"PNG base   : {PNG_OUT_BASE}")
    n_done, n_skip = 0, 0
    for dp, _, files in os.walk(input_base):
        for fn in sorted(files):
            if not fn.lower().endswith(".h5"):
                continue
            src = os.path.join(dp, fn)
            dst = mirror_path(src, input_base, output_base, suffix)

            # resume: skip if augmented exists
            if SKIP_IF_AUGMENTED and os.path.exists(dst):
                print(f"[SKIP] exists: {dst}")
                n_skip += 1
                continue

            png_dir = mirror_png_dir_for_dst(dst, output_base, PNG_OUT_BASE)

            try:
                process_one(src, dst, png_dir=png_dir,
                            gamma_min=gamma_min, gamma_max=gamma_max, print_gamma=print_gamma)
                n_done += 1
            except Exception as e:
                print(f"[FAIL] {src}: {e}")
    print(f"Done. Wrote {n_done} files. Skipped {n_skip} existing.")

if __name__ == "__main__":
    process_tree()