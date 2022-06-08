import numpy as np
from numba import njit

"""Core API: chebsetup, chebinterp, chebder, chebval_scalar, chebval_array"""

@njit
def start(n):
    z = -np.cos(1/(2*n)*np.pi*(2*np.arange(1, n+1)-1))
    Z = _vander(z, n-1)
    P = (2/n)*Z.T.copy()
    P[0, :] /= 2

    return P, z


@njit
def interval(S, a, b):
    P, z = S
    x = _demap(z, a, b)
    return (P, a, b), x


@njit
def quick(n, a, b):
    return interval(start(n), a, b)


@njit
def interp(P, y):
    Pr, a, b = P
    return (Pr @ y, a, b)


@njit
def der(Q):
    Qq, a, b = Q
    Qqder = _der_normalized(Qq)
    return 2/(b-a)*Qqder, a, b


@njit
def val_scalar(Q, x):
    Qq, a, b = Q
    return _val_normalized(_remap(x, a, b), Qq)


@njit
def val_array(Q, x):
    shape = x.shape
    xf = x.ravel()
    yf = np.empty(len(xf))
    for i in range(len(xf)):
        yf[i] = val_scalar(Q, xf[i])
    return yf.reshape(shape)


def val(Q, x):
    """convenience for top-level use, not compatible with Numba and isinstance check is slow!"""
    if isinstance(x, np.ndarray):
        if len(Q) == 3:
            return val_array(Q, x)
        elif len(Q) == 5:
            return val_extrap_array(Q, x)
        else:
            raise ValueError('Chebyshev polynomial should be 3-tuple (or 5-tuple if extrapolated)')
    else:
        if len(Q) == 3:
            return val_scalar(Q, x)
        elif len(Q) == 5:
            return val_extrap_scalar(Q, x)
        else:
            raise ValueError('Chebyshev polynomial should be 3-tuple (or 5-tuple if extrapolated)')


class Chebfunc:
    """Convenience class to represent Chebyshev at top level, not compatible with Numba"""
    def __init__(self, q):
        self.q = q
        self.a = q[1]
        self.b = q[2]
    
    def __call__(self, x):
        return val(self.q, x)

    def der(self):
        return Chebfunc(der(self.q))


"""Less core stuff"""


@njit
def newton(Q, Qder, y, x, err=1E-14):
    for _ in range(30):
        yerror = val_scalar(Q, x) - y
        if abs(yerror) < err:
            return x
        x -= yerror / val_scalar(Qder, x)
    raise ValueError("No convergence in Newton's method after 30 iterations")


@njit
def add(Q1, Q2):
    Qq1, a1, b1 = Q1
    Qq2, a2, b2 = Q2
    
    if a1 != a2 or b1 != b2:
        raise ValueError("Adding polynomials on incompatible intervals")
  
    if len(Qq1) < len(Qq2):
        Qq2 = Qq2.copy()
        Qq2[:len(Qq1)] += Qq1
        return Qq2, a1, b1
    else:
        Qq1 = Qq1.copy()
        Qq1[:len(Qq2)] += Qq2
        return Qq1, a1, b2


@njit
def sub(Q1, Q2):
    Qq1, a1, b1 = Q1
    Qq2, a2, b2 = Q2

    if len(Qq1) < len(Qq2):
        return sub(extend(Q1, len(Qq2)), Q2)
    
    if a1 != a2 or b1 != b2:
        raise ValueError("Subtracting polynomials on incompatible intervals")
  
    Qq1 = Qq1.copy()
    Qq1[:len(Qq2)] -= Qq2
    return Qq1, a1, b2


@njit
def mult_scalar(Q, scalar):
    Qq, a, b = Q
    return scalar*Qq, a, b


@njit
def constant(cons, a, b):
    return np.full(1, float(cons)), a, b


@njit
def extend(Q, n):
    Qq, a, b = Q
    Qnew = np.zeros(n)
    Qnew[:len(Qq)] = Qq
    return Qnew, a, b


@njit
def sup(Q):
    return np.max(np.abs(Q[0]))


"""General interpolation"""


@njit
def interp_general(P, y):
    """Using P from setup_general, interpolate the points y."""
    XV, a, b = P
    return (np.linalg.solve(XV, y), a, b)


@njit
def setup_general(x, a, b):
    """Given arbitrary points x in [a, b], precalculate Vandermonde matrix
    and have it ready for interpolation. Would pre-factor, but doesn't work with Numba"""
    return (_vander(_remap(x, a, b), len(x)-1), a, b)


"""Extrapolation idea"""

@njit
def extrap(q, n=1):
    qq = q[0]
    a = q[1]
    b = q[2]
    pa = np.empty(n+1)
    pb = np.empty(n+1)
    fac = 1
    for deg in range(n+1):
        if deg > 1:
            fac *= deg
        pa[deg] = val_scalar(q, a) / fac
        pb[deg] = val_scalar(q, b) / fac
        if deg < n:
            q = der(q)
    return (qq, a, b, pa, pb)


@njit
def val_extrap_scalar(Q, x):
    Qq, a, b, pa, pb = Q
    if x < a:
        return _polyval(pa, x-a)
    elif x > b:
        return _polyval(pb, x-b)
    else:
        return _val_normalized(_remap(x, a, b), Qq)


@njit
def val_extrap_array(Q, x):
    shape = x.shape
    xf = x.ravel()
    yf = np.empty(len(xf))
    for i in range(len(xf)):
        yf[i] = val_extrap_scalar(Q, xf[i])
    return yf.reshape(shape)


"""Support functions"""


@njit
def _vander(x, deg):
    """
    Return Vandermonde matrix where i,j element is T_j(x_i) for j=0,...,deg, where
    T_j is jth Chebyshev polynomial.

    Adapted from numpy.polynomial.chebyshev.chebvander. Only works for vector x and scalar deg,
    should agree exactly with numpy version in that case.
    """
    v = np.empty((len(x), deg+1))
    for i in range(len(x)):
        xi = x[i]
        v[i, 0] = xi*0 + 1
        if deg > 0:
            x2 = 2*xi
            v[i, 1] = xi
            for j in range(2, deg + 1):
                v[i, j] = v[i, j-1]*x2 - v[i, j-2]
    return v


@njit
def _remap(x, a, b):
    """Map x in [a,b] to z in [-1,1]"""
    return 2/(b-a)*(x-a) - 1


@njit
def _demap(z, a, b):
    """Map z in [-1,1] to x in [a,b]"""
    return (b-a)/2*(z+1) + a


@njit
def _val_normalized(z, c):
    """
    Evaluate a single Chebyshev polynomial c on a scalar z in interval [-1,1].

    Taken from numpy.polynomial.chebyshev.chebval, with extra gunk removed. Only works
    for scalar z and one-dimensional c.
    """
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2*z
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*z


@njit
def _der_normalized(c):
    """
    Given a single Chebyshev polynomial, return the Chebyshev polynomial of its derivative

    Taken from numpy.polynomial.chebyshev.chebder, with extra gunk removed. Only works for one-dimensional c.
    """
    if len(c) == 1:
        return np.array([0.])
    c = c.copy()
    n = len(c) - 1
    der = np.empty(n)
    for j in range(n, 2, -1):
        der[j - 1] = (2*j)*c[j]
        c[j - 2] += (j*c[j])/(j - 2)
    if n > 1:
        der[1] = 4*c[2]
    der[0] = c[1]
    return der


"""Polynomial stuff, not sure if I'll actually use this."""


# @njit
# def chebfrompoly(p, a, b):
#     """SLOW: given coefficients p of polynomial
#     return a Chebyshev representation on [a,b]"""
#     # probably a better algorithm out there...
#     n = len(p)
#     P, x = chebquick(n, a, b)
#     y = np.empty(n)
#     for i in range(n):
#         y[i] = _polyval(p, x[i])
#     return chebinterp(P, y)


@njit
def _polyval(p, x):
    """Same as NumPy, but flip p so constant first"""
    p = p[::-1]
    y = 0
    for i in range(len(p)):
        y = y * x + p[i]
    return y
