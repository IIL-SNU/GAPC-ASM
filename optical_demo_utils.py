# Geometry-Aware Phase Compensation for Sampling-Efficient Angular Spectrum Method
# 2026. 02. 24
# Imaging Intelligence Lab

import torch
import torch.nn.functional as F
import numpy as np

def get_grids(Nx, Ny, px_size, device):
    """
    Returns spatial (XX, YY) and frequency (FX, FY) grids.
    """
    Lx, Ly = Nx * px_size, Ny * px_size
    x = torch.linspace(-Lx/2, Lx/2, Nx, device=device)
    y = torch.linspace(-Ly/2, Ly/2, Ny, device=device)
    XX, YY = torch.meshgrid(x, y, indexing='xy')
    
    fx = torch.fft.fftshift(torch.fft.fftfreq(Nx, d=px_size, device=device))
    fy = torch.fft.fftshift(torch.fft.fftfreq(Ny, d=px_size, device=device))
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    
    return XX, YY, FX, FY

def get_asm_kernel(FX, FY, wavelength, dz, n, x_rad=0.0, y_rad=0.0):
    """
    Handles tilted incident waves by shifting frequency coordinates.
    """
    device = FX.device
    k0 = 2 * torch.pi / wavelength
    two_pi = 2 * torch.pi
    
    fx_0 = (k0 * torch.sin(torch.as_tensor(x_rad, device=device))) / two_pi
    fy_0 = (k0 * torch.sin(torch.as_tensor(y_rad, device=device))) / two_pi
    
    FX_t = FX - fx_0
    FY_t = FY - fy_0
    
    kz_sq = (k0 * n)**2 - (two_pi * FX_t)**2 - (two_pi * FY_t)**2
    kz = torch.sqrt(torch.clamp(kz_sq, min=0.0)).to(torch.complex64)
    
    mask = (kz_sq >= 0).float()
    
    return mask * torch.exp(1j * kz * dz)

def hard_stable_bounded_clip(x, lower, upper):
    """Hard clipping for slicing stability."""
    max_val = (upper - lower)
    if hasattr(max_val, 'item'):
        max_val = max_val.item()
    return torch.clamp(x - lower, min=0.0, max=max_val)

def heightmap_slicer(heightmap, dz_mask, slice_idx, heightmapDirection='front', axial_thickness=None):
    """Generates a slice of the heightmap for Multi-Slice propagation."""
    if axial_thickness is None:
        axial_thickness = heightmap.max()
    heightMask = heightmap - heightmap.min()
    
    numSlices = int(torch.ceil(torch.as_tensor(axial_thickness / dz_mask)).item())
    dz_last = axial_thickness - (numSlices - 1) * dz_mask

    idx = slice_idx if heightmapDirection.lower() in ['front'] else numSlices - 1 - slice_idx

    z_start = idx * dz_mask
    z_actual = dz_mask if idx < numSlices - 1 else dz_last
    z_end = z_start + z_actual

    slice_tensor = hard_stable_bounded_clip(heightMask, z_start, z_end)
        
    return slice_tensor, z_actual

def global_phase_correction(xRad, yRad, n_mat_ch, n_bkgd_ch, h_LPC):
    """Compute OPD term for global phase correction based on provided Snell-aware logic."""
    device = h_LPC.device

    xRad_t = torch.as_tensor(xRad, device=device, dtype=h_LPC.dtype)
    yRad_t = torch.as_tensor(yRad, device=device, dtype=h_LPC.dtype)

    sx1 = torch.sin(xRad_t)
    sy1 = torch.sin(yRad_t)
    sz1_sq = torch.clamp(1.0 - sx1**2 - sy1**2, min=1e-12)
    sz1 = torch.sqrt(sz1_sq)

    n_map = n_mat_ch / n_bkgd_ch
    sx2 = sx1 / n_map
    sy2 = sy1 / n_map

    sz2_sq = torch.clamp(1.0 - sx2**2 - sy2**2, min=1e-12)
    sz2 = torch.sqrt(sz2_sq)

    gpc_scale = 1.0 / sz2
    delta_n = n_mat_ch - n_bkgd_ch
    opd = delta_n * gpc_scale * h_LPC

    return opd

def local_phase_correction(XX, YY, heightmap, skewXRad_array, skewYRad_array, pixel_size, heightmapDirection='front'):
    """
    Local Phase Correction (LPC) by skewing the heightmap.
    This logic implements the piecewise displacement of heightmap coordinates.
    """
    B, C, H, W = skewXRad_array.shape
    DimH, DimW = H * pixel_size, W * pixel_size
    device = heightmap.device

    hm = heightmap.to(device)
    heightmap_batch = hm.view(1, 1, H, W).expand(B, C, H, W)
    if heightmapDirection.lower() in ['front']:
        skewX, skewY = -skewXRad_array, -skewYRad_array
    else:
        skewX, skewY = skewXRad_array, skewYRad_array

    eff_h = (heightmap_batch)

    xs = torch.sin(skewX).to(device)
    ys = torch.sin(skewY).to(device)
    zs = torch.sqrt(torch.clamp(1.0 - xs**2 - ys**2, min=1e-12))
    
    dx = eff_h * (xs / zs)
    dy = eff_h * (ys / zs)

    XX_b = XX.view(1, 1, H, W).to(device)
    YY_b = YY.view(1, 1, H, W).to(device)

    x_new = XX_b + dx
    y_new = YY_b + dy
    x_norm = (2.0 * x_new) / DimW
    y_norm = (2.0 * y_new) / DimH

    grid = torch.stack((x_norm, y_norm), dim=-1)

    heightmap_flat = heightmap_batch.flatten(0, 1).unsqueeze(1).to(torch.float32)
    grid_flat = grid.flatten(0, 1).to(torch.float32)

    tilted_flat = F.grid_sample(
        heightmap_flat, grid_flat,
        mode='bilinear', align_corners=False)

    tilted = tilted_flat.view(B, C, H, W).to(heightmap.dtype)
    nan_mask = torch.isnan(tilted)
    tilted = torch.where(nan_mask, torch.min(hm), tilted)

    return tilted

def propagate_with_cached_kernel(field, H):
    """Base ASM propagation."""
    ef = torch.fft.fftshift(torch.fft.fft2(field, dim=(-2,-1)), dim=(-2,-1))
    ef_prop = ef * H.to(field.dtype)
    return torch.fft.ifft2(torch.fft.ifftshift(ef_prop, dim=(-2,-1)), dim=(-2,-1))

def get_safe_propagation_dist(nx: int, px: float, wl: float, margin: float = 0.9) -> float:
    """
    Calculates the maximum safe distance for single-step ASM propagation.
    """
    L = nx * px
    dz_max = (L * px) / (wl + 1e-12)
    return margin * dz_max

def propagate(field, wavelength, dz, n, FX, FY, px, x_rad=0.0, y_rad=0.0, use_safe_propagation_distance=True):
    """
    Propagates field across dz, using multiple steps if dz exceeds the safe ASM distance.
    """
    Nx = field.shape[-1]
    dz_safe = get_safe_propagation_dist(Nx, px, wavelength) if use_safe_propagation_distance else dz
    
    if dz <= dz_safe:
        H = get_asm_kernel(FX, FY, wavelength, dz, n, x_rad=x_rad, y_rad=y_rad)
        return propagate_with_cached_kernel(field, H)
    else:
        print(f"\t[Warning] dz ({dz:.4f}) > dz_safe ({dz_safe:.4f}). Splitting into {int(np.ceil(dz / dz_safe))} steps.")
        num_steps = int(np.ceil(dz / dz_safe))
        dz_step = dz / num_steps
        H_step = get_asm_kernel(FX, FY, wavelength, dz_step, n, x_rad=x_rad, y_rad=y_rad)
        
        curr_field = field
        for _ in range(num_steps):
            curr_field = propagate_with_cached_kernel(curr_field, H_step)
        return curr_field

def ms_as_engine(field, h_map, dz_val, k0, n_mat, n_bkgd, FX, FY, XX, YY, px_size, heightmapDirection='front', lpc_params=None, angle_params=None):
    """
    Multi-Slice ASM Engine with pre-calculated kernels and piecewise LPC skewing.
    Ensures numerical stability by splitting steps exceeding dz_safe.
    """
    curr_field = field.clone()
    device = field.device
    B, C, H, W = field.shape
    wavelength = 2 * torch.pi / k0
    
    x_rad = angle_params.get('x_rad', 0.0) if angle_params else 0.0
    y_rad = angle_params.get('y_rad', 0.0) if angle_params else 0.0
    
    x_rad_t = torch.as_tensor(x_rad, device=device, dtype=torch.float32).view(1, 1, 1, 1).expand(B, C, H, W)
    y_rad_t = torch.as_tensor(y_rad, device=device, dtype=torch.float32).view(1, 1, 1, 1).expand(B, C, H, W)
    skewX = torch.arcsin(torch.sin(x_rad_t) * (n_bkgd / n_mat))
    skewY = torch.arcsin(torch.sin(y_rad_t) * (n_bkgd / n_mat))
    
    total_thickness = h_map.max()
    num_slices = int(torch.ceil(torch.as_tensor(total_thickness / dz_val)).item())
    
    # Calculate safe sub-stepping for uniform slices
    dz_safe = get_safe_propagation_dist(W, px_size, wavelength)
    num_sub = int(np.ceil(dz_val / dz_safe))
    dz_sub = dz_val / num_sub
    H_sub = get_asm_kernel(FX, FY, wavelength, dz_sub, n_mat, x_rad=x_rad, y_rad=y_rad).to(field.dtype)

    for i in range(num_slices):
        slice_h, dz_eff = heightmap_slicer(h_map, dz_val, i, heightmapDirection, axial_thickness=total_thickness)
        
        h_lpc = slice_h
        if lpc_params is not None:
            h_lpc = local_phase_correction(XX, YY, slice_h, skewX, skewY, px_size, heightmapDirection)
            
        opd = global_phase_correction(x_rad, y_rad, n_mat, n_bkgd, h_lpc)
        
        if heightmapDirection.lower() in ['front']:
            curr_field = curr_field * torch.exp(1j * k0 * opd)
            if abs(dz_eff - dz_val) < 1e-12:
                for _ in range(num_sub): curr_field = propagate_with_cached_kernel(curr_field, H_sub)
            else:
                curr_field = propagate(curr_field, wavelength, dz_eff, n_mat, FX, FY, px_size, x_rad=x_rad, y_rad=y_rad)
        else:
            if abs(dz_eff - dz_val) < 1e-12:
                for _ in range(num_sub): curr_field = propagate_with_cached_kernel(curr_field, H_sub)
            else:
                curr_field = propagate(curr_field, wavelength, dz_eff, n_mat, FX, FY, px_size, x_rad=x_rad, y_rad=y_rad)
            curr_field = curr_field * torch.exp(1j * k0 * opd)               
   
    return curr_field

def ss_as_engine(field, h_map, k0, n_mat, n_bkgd, FX, FY, XX, YY, px_size, heightmapDirection='front', lpc_params=None, angle_params=None):
    """
    Single-Slice ASM engine. Applies GPC/LPC once and propagates across full thickness.
    """
    curr_field = field.clone()
    device = field.device
    B, C, H, W = field.shape
    wavelength = 2 * torch.pi / k0
    thickness = h_map.max()

    x_rad = angle_params.get('x_rad', 0.0) if angle_params else 0.0
    y_rad = angle_params.get('y_rad', 0.0) if angle_params else 0.0
    
    h_lpc = h_map
    if lpc_params is not None:
        x_rad_t = torch.as_tensor(x_rad, device=device, dtype=torch.float32).view(1, 1, 1, 1).expand(B, C, H, W)
        y_rad_t = torch.as_tensor(y_rad, device=device, dtype=torch.float32).view(1, 1, 1, 1).expand(B, C, H, W)
        skewX = torch.arcsin(torch.sin(x_rad_t) * (n_bkgd / n_mat))
        skewY = torch.arcsin(torch.sin(y_rad_t) * (n_bkgd / n_mat))
        h_lpc = local_phase_correction(XX, YY, h_map, skewX, skewY, px_size, heightmapDirection)

    opd = global_phase_correction(x_rad, y_rad, n_mat, n_bkgd, h_lpc)
    mask = torch.exp(1j * k0 * opd)

    if heightmapDirection.lower() in ['front']:
        curr_field = curr_field * mask
        curr_field = propagate(curr_field, wavelength, thickness, n_mat, FX, FY, px_size, x_rad=x_rad, y_rad=y_rad)
    else:
        curr_field = propagate(curr_field, wavelength, thickness, n_mat, FX, FY, px_size, x_rad=x_rad, y_rad=y_rad)
        curr_field = curr_field * mask
        
    return curr_field

def calculate_msssim(img1, img2):
    """Calculates MS-SSIM score with adaptive cropping and padding for stability."""
    from pytorch_msssim import ms_ssim
    if img1.ndim != 2 or img2.ndim != 2: return 0.0
    
    img1_c, img2_c = adaptive_crop_tensors(img1, img2, threshold=1e-4)
    img1_c = torch.as_tensor(img1_c) / (img1_c.max() + 1e-12)
    img2_c = torch.as_tensor(img2_c) / (img2_c.max() + 1e-12)
    
    if img1_c.shape[-2] < 160 or img1_c.shape[-1] < 160:
         pad_h = max(0, 160 - img1_c.shape[-2])
         pad_w = max(0, 160 - img1_c.shape[-1])
         img1_c = torch.nn.functional.pad(img1_c, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
         img2_c = torch.nn.functional.pad(img2_c, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))

    img1_c = img1_c.unsqueeze(0).unsqueeze(0)
    img2_c = img2_c.unsqueeze(0).unsqueeze(0)
    return ms_ssim(img1_c, img2_c, data_range=1.0, size_average=True).item()

def adaptive_crop_tensors(img1, img2, threshold=1e-4):
    """
    Crops img1 and img2 based on the active region of img1 (GT).
    """
    if torch.is_tensor(img1): img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2): img2 = img2.detach().cpu().numpy()

    mask = img1 > (img1.max() * threshold)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(mask):
        return img1, img2

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    r_center = (rmin + rmax) // 2
    c_center = (cmin + cmax) // 2
    half_w = max((rmax - rmin) // 2, (cmax - cmin) // 2) + 10 
    
    H, W = img1.shape
    r_start = max(0, r_center - half_w)
    r_end = min(H, r_center + half_w)
    c_start = max(0, c_center - half_w)
    c_end = min(W, c_center + half_w)

    return img1[r_start:r_end, c_start:c_end], img2[r_start:r_end, c_start:c_end]

def subpixel_shift(intensity, dx_px, dy_px):
    """
    Final PSF centering using Fourier-based subpixel shift.
    """
    Nx, Ny = intensity.shape
    fx = torch.fft.fftfreq(Nx, device=intensity.device)
    fy = torch.fft.fftfreq(Ny, device=intensity.device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    
    I_f = torch.fft.fft2(intensity)
    shift_kernel = torch.exp(-2j * torch.pi * (FX * dx_px + FY * dy_px))
    return torch.fft.ifft2(I_f * shift_kernel).real
