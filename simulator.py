# Author: Stefan Milosevic
"""Main entry point for the reaction-diffusion tumor simulator."""

from fk_1d import fk_solve_1d
from fk_2d import fk_solve_2d, generate_synthetic_brain
from fk_3d import fk_solve_3d
from visualize import compute_growth_metrics, sensitivity_report


if __name__ == '__main__':
    print("=== 1D FK Simulation ===")
    r1 = fk_solve_1d()
    print(f"Wave speed: {r1['theoretical_speed']:.4f}, Doubling: {r1['doubling_time']:.1f}d")

    print("\n=== 2D FK Simulation ===")
    bm, fa = generate_synthetic_brain()
    r2 = fk_solve_2d(fa_map=fa, brain_mask=bm)
    m2 = compute_growth_metrics(r2['growth'])
    print(f"Final volume: {m2['final_volume']:.0f} voxels")

    print("\n=== Sensitivity ===")
    print(sensitivity_report(0.012, 0.1))
    print("\nDone.")
