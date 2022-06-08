import numpy as np


def J_from_F(F, T=300):
    """From fake news matrix F (possibly smaller than T*T), build Jacobian J of size T*T"""

    if T < len(F):
        raise ValueError(f"T={T} must be weakly larger than existing size of F, {len(F)}")

    J = np.zeros((T, T))
    J[:len(F), :len(F)] = F
    for t in range(1, J.shape[1]):
        J[1:, t] += J[:-1, t - 1]
    return J


def Jreal_from_Fnom(Fnom, T=1000):
    """From nominal fake news matrix get T*T real Jacobian. This calculation only works well with large T."""
    return Jreal_from_Jnom(J_from_F(Fnom, T))


def Jreal_from_Jnom(Jnom):
    Jreal = np.linalg.solve(np.eye(len(Jnom)) - Jnom, Jnom)
    Jreal[1:, :] = Jreal[1:, :] - Jreal[:-1, :]
    return Jreal


def Fnom_for_td(Phi, beta):
    """Build fake news matrix for a time-dependent pricer with survival curve Phi and discount beta"""
    columns = beta**np.arange(len(Phi))*Phi
    columns /= columns.sum()

    rows = Phi / Phi.sum()
    return np.outer(rows, columns)


def Jnom_for_td(Phi, beta, T=300):
    return J_from_F(Fnom_for_td(Phi, beta), T)


def Jreal_for_td(Phi, beta, T=1000):
    return Jreal_from_Fnom(Fnom_for_td(Phi, beta), T)


