"""2D Fisher-Kolmogorov solver with anisotropic diffusion support."""

import numpy as np


def fk_solve_2d(nx=128, ny=128, T=150.0, dt=0.1, rho=0.012, D_white=0.1,
                D_gray=0.01, brain_mask=None, fa_map=None, tumor_seed=None):
    """Solve the 2D FK equation with optional FA-weighted diffusion."""
    nt = int(T / dt)
    if brain_mask is None:
        y, x = np.ogrid[-1:1:ny*1j, -1:1:nx*1j]
        brain_mask = ((x**2 + y**2) < 0.85).astype(np.float64)
    D_field = np.ones((ny, nx)) * D_gray
    if fa_map is not None:
        fa_norm = np.clip(fa_map / (np.max(fa_map) + 1e-8), 0, 1)
        D_field = D_gray + (D_white - D_gray) * fa_norm
    D_field *= brain_mask
    u = np.zeros((ny, nx), dtype=np.float64)
    if tumor_seed is None:
        cx, cy, r = nx // 2, ny // 2, 5
    else:
        cx, cy, r = tumor_seed
    yy, xx = np.ogrid[0:ny, 0:nx]
    u[((xx - cx)**2 + (yy - cy)**2) < r**2] = 0.8
    u *= brain_mask
    snapshots = []
    growth = []
    save_interval = max(1, nt // 50)
    for step in range(nt + 1):
        growth.append(float(np.sum(u > 0.1)))
        if step % save_interval == 0:
            snapshots.append(u.copy())
        if step < nt:
            u_pad = np.pad(u, 1, mode='constant')
            lap = (u_pad[:-2, 1:-1] + u_pad[2:, 1:-1] +
                   u_pad[1:-1, :-2] + u_pad[1:-1, 2:] - 4 * u)
            u = u + dt * (D_field * lap + rho * u * (1 - u))
            u *= brain_mask
            u = np.clip(u, 0, 1)
    return {'snapshots': snapshots, 'growth': growth,
            'brain_mask': brain_mask, 'D_field': D_field}


def generate_synthetic_brain(nx=128, ny=128):
    """Generate synthetic brain with white/gray matter regions."""
    y, x = np.ogrid[-1:1:ny*1j, -1:1:nx*1j]
    brain_mask = ((x / 0.9)**2 + (y / 0.75)**2 < 1).astype(np.float64)
    r = np.sqrt(x**2 + y**2)
    fa_map = np.clip(0.6 - r * 0.5, 0.05, 0.7)
    cc_mask = (np.abs(y) < 0.08) & (np.abs(x) < 0.6)
    fa_map[cc_mask] = 0.65
    fa_map *= brain_mask
    return brain_mask, fa_map
