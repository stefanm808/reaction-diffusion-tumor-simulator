"""Utility functions for the tumor growth simulator."""

import numpy as np


def wave_speed(rho, D):
    """Theoretical FK wave front speed."""
    return 2 * np.sqrt(D * rho)


def doubling_time(rho):
    """Tumor volume doubling time from proliferation rate."""
    return np.log(2) / rho if rho > 0 else float('inf')


def dice_coefficient(a, b):
    """Dice similarity coefficient between two binary masks."""
    return 2 * np.sum(a * b) / (np.sum(a) + np.sum(b) + 1e-8)


def tumor_volume_ml(voxels, voxel_mm=1.0):
    """Convert voxel count to volume in milliliters."""
    return voxels * voxel_mm**3 / 1000.0


def cfl_check(D, dx, dt):
    """Check CFL stability condition for explicit scheme."""
    cfl = D * dt / dx**2
    return {'cfl': cfl, 'stable': cfl <= 0.5}
