"""1D Fisher-Kolmogorov solver for tumor growth."""

import numpy as np


def fk_solve_1d(L=100.0, nx=500, T=200.0, dt=0.05, rho=0.012, D=0.1,
                ic_center=None, ic_width=3.0):
    """Solve the 1D FK equation using explicit finite differences."""
    dx = L / nx
    x = np.linspace(0, L, nx)
    nt = int(T / dt)
    if ic_center is None:
        ic_center = L / 2.0
    cfl = D * dt / dx**2
    if cfl > 0.5:
        dt = 0.4 * dx**2 / D
        nt = int(T / dt)
    u = np.exp(-((x - ic_center)**2) / (2 * ic_width**2))
    u = np.clip(u, 0, 1)
    u_history = [u.copy()]
    volumes = [float(np.sum(u > 0.1) * dx)]
    save_interval = max(1, nt // 100)
    for step in range(1, nt + 1):
        u_pad = np.pad(u, 1, mode='edge')
        laplacian = (u_pad[2:] - 2 * u_pad[1:-1] + u_pad[:-2]) / dx**2
        du = D * laplacian + rho * u * (1 - u)
        u = u + dt * du
        u = np.clip(u, 0, 1)
        if step % save_interval == 0:
            u_history.append(u.copy())
            volumes.append(float(np.sum(u > 0.1) * dx))
    return {
        'x': x, 'u_history': u_history, 'volumes': volumes,
        'theoretical_speed': 2 * np.sqrt(D * rho),
        'doubling_time': np.log(2) / rho if rho > 0 else float('inf'),
    }


def parameter_sweep_1d(rho_range=(0.005, 0.025), D_range=(0.05, 0.25),
                        n_rho=8, n_D=8, T=100.0):
    """Sweep rho and D to map tumor growth behavior."""
    rhos = np.linspace(rho_range[0], rho_range[1], n_rho)
    Ds = np.linspace(D_range[0], D_range[1], n_D)
    final_volumes = np.zeros((n_rho, n_D))
    wave_speeds = np.zeros((n_rho, n_D))
    for i, rho in enumerate(rhos):
        for j, D in enumerate(Ds):
            result = fk_solve_1d(rho=rho, D=D, T=T, nx=200)
            final_volumes[i, j] = result['volumes'][-1]
            wave_speeds[i, j] = result['theoretical_speed']
    return {'rhos': rhos, 'Ds': Ds, 'final_volumes': final_volumes, 'wave_speeds': wave_speeds}


if __name__ == '__main__':
    result = fk_solve_1d()
    print(f"Wave speed: {result['theoretical_speed']:.4f} mm/day")
    print(f"Doubling time: {result['doubling_time']:.1f} days")
