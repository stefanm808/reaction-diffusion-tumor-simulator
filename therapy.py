"""Therapy response simulation with LQ radiation model."""

import numpy as np

LQ_ALPHA = 0.035
LQ_BETA = 0.0035
RT_DOSE = 2.0
TMZ_KILL = 0.005


def simulate_therapy(growth_baseline, n_mc=50):
    """Simulate therapy regimens against untreated baseline."""
    n = len(growth_baseline)
    results = {}
    for name, use_rt, use_tmz in [
        ('Untreated', False, False),
        ('Chemoradiation', True, True),
        ('Radiation Only', True, False),
    ]:
        curves = []
        for _ in range(n_mc):
            a = max(LQ_ALPHA * (1 + np.random.randn() * 0.15), 0.01)
            b = max(LQ_BETA * (1 + np.random.randn() * 0.15), 0.001)
            t_k = max(TMZ_KILL * (1 + np.random.randn() * 0.15), 0.001)
            curve = [growth_baseline[0]]
            for t in range(1, n):
                base = growth_baseline[t] / max(growth_baseline[t - 1], 1e-10)
                val = curve[-1] * base
                if use_rt and t <= 42 and t % 7 not in [6, 0]:
                    val *= np.exp(-(a * RT_DOSE + b * RT_DOSE**2))
                if use_tmz and t <= 72:
                    val *= np.exp(-t_k)
                curve.append(max(val, curve[0] * 0.01))
            curves.append(curve)
        arr = np.array(curves)
        results[name] = {
            'median': np.median(arr, axis=0).tolist(),
            'p5': np.percentile(arr, 2.5, axis=0).tolist(),
            'p95': np.percentile(arr, 97.5, axis=0).tolist(),
        }
    return results
