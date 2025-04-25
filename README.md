# Reaction-Diffusion Tumor Growth Simulator

A Python implementation of the Fisher-Kolmogorov reaction-diffusion model for simulating glioma invasion in the brain.

## The Model

The FK equation describes how tumor cells proliferate and migrate through brain tissue:

    du/dt = D * nabla^2(u) + rho * u * (1 - u)

Where:
- u(x,t) is the normalized tumor cell density (0 to 1)
- D is the diffusion coefficient (how fast cells spread)
- rho is the proliferation rate (how fast cells divide)

## Features

- 1D, 2D, and 3D FK solvers with configurable parameters
- Anisotropic diffusion using DTI fractional anisotropy maps
- Parameter sensitivity analysis across clinically relevant ranges
- Therapy response simulation (LQ radiation + chemotherapy)
- Survival estimation from model parameters and molecular markers

## Clinical Context

Glioblastoma (GBM) is the most aggressive primary brain tumor with ~15 month median survival. Tumor cells infiltrate beyond the visible MRI margin along white matter tracts. The FK model captures this invisible invasion, which is critical for radiation therapy planning.

## Quick Start

    pip install numpy scipy matplotlib
    python simulator.py

## Usage

    from fk_1d import fk_solve_1d
    result = fk_solve_1d(rho=0.012, D=0.1, T=200)
    print(f"Wave speed: {result['theoretical_speed']:.4f} mm/day")

    from survival import estimate_survival
    surv = estimate_survival(rho=0.012, D=0.1, idh='WT', mgmt_meth=True)
    print(f"Median OS: {surv['median_os']} months")

## References

- Fisher (1937). The wave of advance of advantageous genes. Annals of Eugenics.
- Swanson et al. (2000). A quantitative model for differential motility of gliomas.
- Harpold et al. (2007). Evolution of mathematical modeling of glioma proliferation and invasion.
- Stupp et al. (2005). Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma.

## Contributing

Contributions welcome. Please open an issue first to discuss proposed changes.

## License

MIT License
