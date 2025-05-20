"""Visualization utilities and growth metric calculations."""

import numpy as np


def compute_growth_metrics(growth, dt=0.1):
    """Compute clinically relevant metrics from growth curve."""
    volumes = np.array(growth, dtype=np.float64)
    days = np.arange(len(volumes)) * dt
    if volumes[0] > 0 and volumes[-1] > volumes[0]:
        ratio = volumes[-1] / volumes[0]
        doublings = np.log2(ratio)
        doubling_time = days[-1] / doublings if doublings > 0 else float('inf')
    else:
        doubling_time = float('inf')
    return {
        'doubling_time_empirical': doubling_time,
        'initial_volume': float(volumes[0]),
        'final_volume': float(volumes[-1]),
    }


def sensitivity_report(rho_base, D_base, perturbation=0.2):
    """Generate parameter sensitivity report."""
    wave_base = 2 * np.sqrt(D_base * rho_base)
    lines = [f"Baseline: rho={rho_base}, D={D_base}, wave={wave_base:.4f}"]
    for name, rho, D in [
        ("rho+20%", rho_base * 1.2, D_base),
        ("rho-20%", rho_base * 0.8, D_base),
        ("D+20%", rho_base, D_base * 1.2),
        ("D-20%", rho_base, D_base * 0.8),
    ]:
        ws = 2 * np.sqrt(D * rho)
        lines.append(f"  {name}: wave={ws:.4f} ({(ws / wave_base - 1) * 100:+.1f}%)")
    return "\n".join(lines)
