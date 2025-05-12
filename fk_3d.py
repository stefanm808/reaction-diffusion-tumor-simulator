"""3D Fisher-Kolmogorov solver for volumetric tumor growth."""

import numpy as np


def fk_solve_3d(shape=(64, 64, 64), T=100.0, dt=0.2, rho=0.012,
                D_white=0.1, D_gray=0.01, brain_mask=None,
                fa_volume=None, tumor_seed=None):
    """Solve the 3D FK equation on a brain volume."""
    nz, ny, nx = shape
    nt = int(T / dt)
    if brain_mask is None:
        z, y, x = np.ogrid[-1:1:nz*1j, -1:1:ny*1j, -1:1:nx*1j]
        brain_mask = ((x**2 + y**2 + z**2) < 0.8).astype(np.float64)
    D_field = np.ones(shape) * D_gray
    if fa_volume is not None:
        fa_norm = np.clip(fa_volume / (np.max(fa_volume) + 1e-8), 0, 1)
        D_field = D_gray + (D_white - D_gray) * fa_norm
    D_field *= brain_mask
    u = np.zeros(shape, dtype=np.float64)
    if tumor_seed is None:
        cx, cy, cz, r = nx // 2, ny // 2, nz // 2, 3
    else:
        cx, cy, cz, r = tumor_seed
    zz, yy, xx = np.ogrid[0:nz, 0:ny, 0:nx]
    u[((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2) < r**2] = 0.8
    u *= brain_mask
    growth = []
    for step in range(nt + 1):
        growth.append(float(np.sum(u > 0.1)))
        if step < nt:
            u_pad = np.pad(u, 1, mode='constant')
            lap = (u_pad[:-2, 1:-1, 1:-1] + u_pad[2:, 1:-1, 1:-1] +
                   u_pad[1:-1, :-2, 1:-1] + u_pad[1:-1, 2:, 1:-1] +
                   u_pad[1:-1, 1:-1, :-2] + u_pad[1:-1, 1:-1, 2:] - 6 * u)
            u = u + dt * (D_field * lap + rho * u * (1 - u))
            u *= brain_mask
            u = np.clip(u, 0, 1)
    return {
        'u_final': u, 'growth': growth,
        'wave_speed': 2 * np.sqrt(D_white * rho),
        'doubling_time': np.log(2) / rho if rho > 0 else float('inf'),
    }
