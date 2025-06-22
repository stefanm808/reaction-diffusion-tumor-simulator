"""Survival estimation from FK parameters and molecular markers."""

import numpy as np


def estimate_survival(rho, D, idh='WT', mgmt_meth=False, age=55, eor='GTR'):
    """Estimate overall survival from model parameters and clinical factors."""
    base_os = 15.0
    if idh == 'Mut':
        base_os += 20
    if mgmt_meth:
        base_os += 6
    if eor == 'GTR':
        base_os += 3
    elif eor == 'Biopsy':
        base_os -= 3
    if age > 65:
        base_os -= 4
    elif age < 40:
        base_os += 3
    months = np.arange(0, 37)
    km = np.exp(-np.log(2) / max(base_os, 1) * months)
    return {
        'median_os': round(base_os, 1),
        'km_months': months.tolist(),
        'km_survival': km.tolist(),
    }


if __name__ == '__main__':
    s1 = estimate_survival(0.015, 0.1, idh='WT', mgmt_meth=False, age=62)
    print(f"IDH-WT patient: {s1['median_os']} months")
    s2 = estimate_survival(0.008, 0.05, idh='Mut', mgmt_meth=True, age=38, eor='GTR')
    print(f"IDH-Mut patient: {s2['median_os']} months")
