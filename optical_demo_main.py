# Geometry-Aware Phase Compensation for Sampling-Efficient Angular Spectrum Method
# 2026. 02. 24
# Imaging Intelligence Lab

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from optical_demo_utils import (
    get_grids, get_asm_kernel, ms_as_engine, ss_as_engine, subpixel_shift,
    propagate, adaptive_crop_tensors, calculate_msssim
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mm, um, nm = 1e-3, 1e-6, 1e-9
use_safe_propagation_distance = False
Grid_Window = 1.5 * mm 
PxSize = 1.0 * um
Nx = Ny = int(Grid_Window / PxSize)

Lens_Dia = 1.0  * mm
ROC = 2.0 * mm 
WD = 4.0* mm
lens_thick = 100 * um  
n_mat = 1.54
n_bkgd = 1.0

Illum_Dia = 0.7 * Lens_Dia 
wavelength = 550 * nm
k0 = 2 * torch.pi / wavelength

angle_deg_x, angle_deg_y = 20, 0
angle_rad_x, angle_rad_y = np.deg2rad(angle_deg_x), np.deg2rad(angle_deg_y)
nx_inc = np.sin(angle_rad_x)
ny_inc = np.sin(angle_rad_y)

dz_gt = 1.0 * um
dz_coarse = 50.0 * um

Crop_window = 300*um

print(f"--- Optical Simulation Demo ---")
print(f"Grid: {Nx}x{Ny} | Px: {PxSize/um:.1f}um | Window: {Grid_Window/mm:.2f}mm")
print(f"Lens Diameter: {Lens_Dia/mm:.2f}mm | Illum Diameter: {Illum_Dia/mm:.2f}mm")
print(f"Angle: {angle_deg_x} deg, {angle_deg_y} deg | GT dz: {dz_gt/um:.1f} um | Coarse dz: {dz_coarse/um:.1f} um")

XX, YY, FX, FY = get_grids(Nx, Ny, PxSize, device)

RR = torch.sqrt(XX**2 + YY**2)
Illum_ap = (RR <= Illum_Dia*0.5).float()
input_field = Illum_ap.unsqueeze(0).unsqueeze(0).to(torch.complex64)

lens_ap = (RR <= Lens_Dia*0.5).float()
h_map_raw = ROC - torch.sqrt(torch.clamp(ROC**2 - RR**2, min=0.0))
h_map = (torch.ones_like(RR) * lens_thick) - h_map_raw
h_map = h_map * lens_ap 
h_map = h_map - torch.min(h_map) 

def plot_geometry():
    """Visualizes the lens heightmap and cross-section profile."""
    h_cpu = h_map.cpu().numpy()
    Illum_cpu = Illum_ap.cpu().numpy()
    center_row = h_cpu[Ny//2, :]
    x_axis = np.linspace(-Grid_Window/2, Grid_Window/2, Nx) / mm
    
    fig_geo, ax_geo = plt.subplots(1, 2, figsize=(14, 6))
    
    extent = [-Grid_Window/2/mm, Grid_Window/2/mm, -Grid_Window/2/mm, Grid_Window/2/mm]
    im = ax_geo[0].imshow(h_cpu / um, extent=extent, cmap='viridis', alpha=0.9)
    ax_geo[0].contour(Illum_cpu[0,0] if Illum_cpu.ndim==4 else Illum_cpu, levels=[0.5], 
                      extent=extent, colors='orange', linestyles='--')
    
    ax_geo[0].set_title("Heightmap & Illum Aperture", fontsize=14, fontweight='bold')
    ax_geo[0].set_xlabel("x [mm]")
    ax_geo[0].set_ylabel("y [mm]")
    plt.colorbar(im, ax=ax_geo[0], label="Height [um]")
    
    ax_geo[1].plot(x_axis, center_row / um, color='blue', linewidth=2.5, label='Lens Profile')
    ax_geo[1].axvline(-Lens_Dia*0.5/mm, color='red', linestyle='--', alpha=0.6, label='Lens Edge')
    ax_geo[1].axvline(Lens_Dia*0.5/mm, color='red', linestyle='--', alpha=0.6)
    ax_geo[1].axvline(-Illum_Dia*0.5/mm, color='orange', linestyle=':', alpha=0.8, label='Illum Edge')
    ax_geo[1].axvline(Illum_Dia*0.5/mm, color='orange', linestyle=':', alpha=0.8)
    
    ax_geo[1].axvline(-Grid_Window*0.5/mm, color='black', linestyle='-', alpha=0.3, label='Grid Edge')
    ax_geo[1].axvline(Grid_Window*0.5/mm, color='black', linestyle='-', alpha=0.3)
    
    ax_geo[1].set_title("Integrated Aperture Profile", fontsize=14, fontweight='bold')
    ax_geo[1].set_xlabel("x [mm]")
    ax_geo[1].set_ylabel("Thickness [um]")
    ax_geo[1].legend(loc='upper right', fontsize='small')
    ax_geo[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("optical_demo_geometry.png", dpi=300)
    print(f"Geometry plot saved to optical_demo_geometry.png")

plot_geometry()

lpc_params = {
    'XX': XX, 'YY': YY, 
    'nx': torch.tensor(nx_inc, device=device), 
    'ny': torch.tensor(ny_inc, device=device)
}

angle_params = {'x_rad': angle_rad_x, 'y_rad': angle_rad_y}

def run_simulation(method_name, direction, dz=None):
    """Executes a single simulation run for a given method and direction."""
    print(f"Running {method_name} ({direction})...")
    torch.cuda.empty_cache()

    if "MS-AS" in method_name:
        is_lpc = "+LPC" in method_name
        field = ms_as_engine(input_field, h_map, dz, k0, n_mat, n_bkgd, FX, FY, 
                            XX, YY, px_size=PxSize, heightmapDirection=direction, 
                            lpc_params=lpc_params if is_lpc else None, 
                            angle_params=angle_params)
    elif "SS-AS" in method_name:
        is_lpc = "+LPC" in method_name
        field = ss_as_engine(input_field, h_map, k0, n_mat, n_bkgd, FX, FY,
                                  XX, YY, px_size=PxSize, heightmapDirection=direction, 
                                  lpc_params=lpc_params if is_lpc else None, 
                                  angle_params=angle_params)
    
    final_field = propagate(field, wavelength, WD, n_bkgd, FX, FY, PxSize, x_rad=angle_rad_x, y_rad=angle_rad_y, use_safe_propagation_distance=use_safe_propagation_distance)
    intensity = torch.abs(final_field[0, 0])**2
    
    theta_mat_x = np.arcsin(np.sin(angle_rad_x)/n_mat)
    theta_mat_y = np.arcsin(np.sin(angle_rad_y)/n_mat)
    shift_x = WD * np.tan(angle_rad_x) + lens_thick * np.tan(theta_mat_x)
    shift_y = WD * np.tan(angle_rad_y) + lens_thick * np.tan(theta_mat_y)

    intensity_centered = subpixel_shift(intensity, shift_x/PxSize, shift_y/PxSize)
     
    return intensity_centered.detach().cpu()

methods = [
    ("MS-AS (GT, 1um)", dz_gt),
    ("MS-AS (Low, 50um)", dz_coarse),
    ("MS-AS+LPC (Low, 50um)", dz_coarse),
    ("SS-AS (Single)", None),
    ("SS-AS+LPC (Single)", None)
]

directions = ['front', 'back']

results = {}
for name, dz in methods:
    for d in directions:
        key = f"{name} [{d}]"
        results[key] = run_simulation(name, d, dz)

metrics = {}
for name, _ in methods:
    for d in directions:
        key = f"{name} [{d}]"
        ref_key = f"MS-AS (GT, 1um) [{d}]"
        metrics[key] = calculate_msssim(results[ref_key], results[key])

fig, axes = plt.subplots(2, 5, figsize=(25, 12))

c = Nx // 2
r = int(Crop_window / (2 * PxSize))
crop_size_um = Crop_window / um

for col_idx, (name, _) in enumerate(methods):
    for row_idx, d in enumerate(directions):
        ax = axes[row_idx, col_idx]
        key = f"{name} [{d}]"
        crop = results[key][c-r:c+r, c-r:c+r].numpy()
        
        H, W = crop.shape
        yy_c = np.linspace(-H/2, H/2, H) * (PxSize / um)
        xx_c = np.linspace(-W/2, W/2, W) * (PxSize / um)
        XX_c, YY_c = np.meshgrid(xx_c, yy_c)
        total_i = crop.sum() + 1e-12
        cx = (crop * XX_c).sum() / total_i
        cy = (crop * YY_c).sum() / total_i
        
        crop_disp = crop / (crop.max() + 1e-12)
        extent = [-crop_size_um*0.5, crop_size_um*0.5, -crop_size_um*0.5, crop_size_um*0.5]
        ax.imshow(crop_disp, cmap='inferno', extent=extent)
        
        m_score = metrics[key]
        if col_idx == 0:
            ax.set_ylabel(f"Dir: {d}", fontsize=14, fontweight='bold')
        
        if row_idx == 0:
            ax.set_title(f"{name}", fontsize=12, fontweight='bold')

        ax.text(0.05, 0.95, f"SSIM: {m_score:.4f}\n({cx:.1f}, {cy:.1f}) um", 
                transform=ax.transAxes, color='white', verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        ax.grid(True, alpha=0.3, linestyle='--')

plt.suptitle(f"2x5 PSF Fidelity Study: Front vs Back Orientations\n"
             f"(Angle: {angle_deg_x} deg, {angle_deg_y} deg, WD: {WD/mm:.1f}mm, Grid: {Grid_Window/mm:.1f}mm)", 
             fontsize=18, fontweight='bold', y=0.97)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("optical_demo_results.png", dpi=300)
print(f"Done! Results saved to optical_demo_results.png")
