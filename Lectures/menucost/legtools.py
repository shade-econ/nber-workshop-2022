from numpy.polynomial import legendre
from numba import njit

def start(n):
    return legendre.leggauss(n)

@njit
def interval(S, a, b):
    z, wnorm = S
    x = _demap(z, a, b)
    w = (b-a)/2*wnorm
    return w, x
    
def quick(n, a, b):
    return interval(start(n), a, b)

@njit
def _demap(z, a, b):
    """Map z in [-1,1] to x in [a,b]"""
    return (b-a)/2*(z+1) + a
