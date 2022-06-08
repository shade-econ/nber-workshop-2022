"""Code for discrete-time menu cost model where price gaps have N(mu, sigma^2) innovations,
can be reset at cost 'c' or with free adjustments that arrive with probability 'lambda', and
where pricer minimizes discounted expected flow loss function (x-ideal)^2/2, where 'x' is price gap,
'ideal' is static ideal point, and discount rate is 'beta'.

We generally refer to the price gap reset point as 'xstar', the Ss bounds as [x_l, x_h],
the probability mass at the reset point as 'p', the density of prices on [x_l, x_h]
as f(x), the beginning-of-period value function (prior to price gap innovations and resets)
as W(x), and the end-of-period value function (including today's loss function) as V(x).

Bellman equations are 

W_t(x) = (1-lambda)*E[V_t(x') * 1(x' in [x_l, x_h])] + (1-lambda)*(V_t(xstar)+c)*Pr[x' not in [x_l, x_h]]
        + lambda*V(xstar)
        
where x' is x plus the N(mu,sigma^2) innovation, and

V_t(x) = (x-ideal_t)^2/2 + beta*W_(t+1)(x)

where xstar, x_l, and x_h are chosen each period to minimize W_t(x), giving rise to the FOCs:

V_t(x_lt) = V_t(xstar_t) + c        (1)
V_t(x_ht) = V_t(xstar_t) + c        (2)
V_t'(xstar_t) = 0                   (3)
"""

import numpy as np
from numba import njit
import math

from . import jacobian, chebtools as cheb, legtools as leg

def solve(c, ideal, beta, lamb, mu, sigma, a, b, T_surv=50, T_Jnom=100, T_Jreal=1000):
    """Convenience function: For a given menu cost model calibration, return nominal Jacobian,
    real Jacobian, policy, mass point and distribution, survival curves and weights for all three
    implied TD pricers, and their corresponding nominal Jacobians."""
    # solve for steady-state policy and distribution
    _, *policy, p, qf, _ = steady_state(c, ideal, beta, lamb, mu, sigma, a, b, verbose=False)
    policy = tuple(policy)
    dist = (p, cheb.Chebfunc(qf))

    # calculate E^s(x) functions on [x_l, x_h] and use them to get equivalent three TD-pricer representation
    Es = get_Es(policy, lamb, mu, sigma, T_surv)
    weights, Phis = get_equivalent_td(policy, (p, qf, *policy), Es)
    weights, Phis = tuple(weights), tuple(Phis.T) # make these tuples for xstar, x_l, x_h

    Fnoms = [jacobian.Fnom_for_td(Phi, beta) for Phi in Phis]
    Fnom = sum(weight*F for weight, F in zip(weights, Fnoms))

    Jnoms = [jacobian.J_from_F(F, T_Jnom) for F in Fnoms]
    Jnom = sum(weight*J for weight, J in zip(weights, Jnoms))

    Jreal = jacobian.Jreal_from_Fnom(Fnom, T_Jreal)

    return Jnom, Jreal, policy, dist, weights, Es, Phis, Jnoms


"""
Core routines needed for steady state: 
    - backward iteration
    - steady-state policy
    - forward iteration
    - steady-state distribution
    - aggregation
    - overall steady state

Note that we frequently pass around data in tuples (e.g. 'setup' for Chebyshev and quadrature objects)
as a more easily njit-compatible alternative to classes.

Also, we make use of the 'cheb' and 'leg' modules for Chebyshev interpolation and Gauss-Legendre quadrature.
Importantly, 'cheb' represents degree-n Chebyshev polynomials on an interval [a, b] as 3-tuples qW = (QW, a, b),
where QW is a length-(n+1) vector giving the coefficients; sometimes we access these coefficients directly.

When we represent a continuous function by a Chebyshev polynomial, we precede it with q, e.g. 'qf' for density f.
"""


@njit
def backward_iterate(qW, c, ideal, beta, lamb, mu, sigma, setup):
    # extract stuff for Chebyshev / quadrature that is constant across iteration
    L, P, x, P_quadratic, x_quadratic = setup
    
    # loss today is 1/2*(x-ideal)**2, use brute force to make this Chebyshev polynomial
    qLoss = cheb.interp(P_quadratic, 1/2*(x_quadratic-ideal)**2)
    
    # get V = loss + beta * W
    qV = cheb.add(qLoss, cheb.mult_scalar(qW, beta))
    
    # solve FOCs to find optimal reset point and Ss bands, and also V(xstar)
    xstar, x_l, x_h, Vxstar = get_optimal_reset_and_thresholds(qV, c)

    # get W(x) at each chebyshev node in x
    W = np.empty_like(x)
    for i in range(len(x)):
        # mean of x' (after innovation) is beginning-of-period x plus mu
        exp_xprime = x[i] + mu
        
        # contribution to expected value from no-reset case, i.e. no free adj and x' in [x_l, x_h]
        no_reset = (1-lamb) * integrate_within_interval(qV, x_l, x_h, exp_xprime, sigma, L)
        
        # contribution to expected value from paid reset case, i.e. no free adj and x' outside [x_l, x_h]
        reset_paid = (1-lamb) * (1 - probability_within_interval(x_l, x_h, exp_xprime, sigma)) * (Vxstar + c)
        
        # contribution from free reset case
        reset_free = lamb * Vxstar

        W[i] = no_reset + reset_paid + reset_free
    
    qW = cheb.interp(P, W)
    
    return qW, xstar, x_l, x_h


def steady_state_policy(c, ideal, beta, lamb, mu, sigma, a, b, Cheb, Leg, tol=1E-13, verbose=True, max_it=1000):
    """Calculate steady-state policy (xstar, x_l, x_h) and beginning-of-period value function W(x), given
    parameters, objects (Cheb, Leg) for interpolation and quadrature, and an interval [a, b] on which to
    obtain W(x) (ideally tight around x_l, x_h)."""

    # map Cheb and Leg to interval [a, b], also make simple Chebyshev for quadratic, combine in 'setup'
    P_quadratic, x_quadratic = cheb.quick(3, a, b)
    P, x = cheb.interval(Cheb, a, b)
    setup = (Leg, P, x, P_quadratic, x_quadratic)

    # initialize beginning-of-period value function at constant 0 on [a, b]
    qWold = cheb.constant(0, a, b)
    
    # iterate until convergence
    for it in range(max_it):
        qW, xstar, x_l, x_h = backward_iterate(qWold, c, ideal, beta, lamb, mu, sigma, setup)

        # iterate until max difference in Chebyshev coefficients excluding first (constant) is below tol
        # we ignore constant because it is irrelevant for all decisions but takes longer to converge
        if it > 0 and np.max(np.abs(cheb.sub(qW, qWold)[0][1:])) < tol:
            if verbose:
                print(f'Backward iteration: convergence in {it} iterations!')
            return qW, xstar, x_l, x_h
        
        qWold = qW
        
    raise ValueError(f'Did not converge to tolerance {tol} even after {max_it} iterations!')


@njit
def forward_iterate(dist_lag, policy, params):
    """Taking yesterday's end-of-period distribution, which is a tuple (p, f, xstar, x_l, x_h),
    and today's policy, which is a tuple (xstar, x_l, x_h), update to today's end-of-period distribution."""
    # unpack yesterday's distributon, today's policy, and constant parameters
    p_lag, qf_lag, xstar_lag, x_l_lag, x_h_lag = dist_lag
    xstar, x_l, x_h = policy
    lamb, mu, sigma, Cheb, Leg = params
    
    # iterate forward to get today's mass at reset point and density elsewhere
    p = p_next(p_lag, qf_lag, xstar_lag, x_l_lag, x_h_lag, x_l, x_h, lamb, mu, sigma, Leg)
    qf = f_next(p_lag, qf_lag, xstar_lag, x_l_lag, x_h_lag, x_l, x_h, lamb, mu, sigma, Cheb, Leg)
    
    # pack this into a tuple reprsenting today's end-of-period distribution
    dist = (p, qf, xstar, x_l, x_h)
    return dist


@njit
def steady_state_dist(xstar, x_l, x_h, lamb, mu, sigma, Cheb, Leg, tol=1E-9, verbose=True, max_it=1000):
    """Given steady-state policy (xstar, x_l, x_h), parameters for price gap innovations N(mu, sigma^2),
    and objects (Cheb, Leg) for interpolation and quadrature, find steady-state mass 'p' at reset point
    xstar, and steady-state density 'f' on [x_l, x_h], return as tuple (p, f, xstar, x_l, x_h) with policy."""
    # initialize constant parameters and policy tuple
    params = (lamb, mu, sigma, Cheb, Leg)
    policy = (xstar, x_l, x_h)
    
    # now initialize initial guess: uniform on x_l to x_h, no mass point
    qf = cheb.constant(1/(x_h-x_l), x_l, x_h)
    p = 0.
    dist_old = (p, qf, xstar, x_l, x_h)

    # iterate until convergence
    for it in range(max_it):
        dist = forward_iterate(dist_old, policy, params)

        # require both change in mass p and sup change in Chebyshev coeff for density f to be below 'tol'
        if np.abs(dist[0] - dist_old[0]) < tol and cheb.sup(cheb.sub(dist[1], dist_old[1])) < tol:
            if verbose:
                print('Forward iteration: convergence in the following number of iterations')
                print(it)
            return dist
    
        dist_old = dist
        
    raise ValueError('Distribution did not converge to tolerance even after maximum iterations!')


@njit
def aggregate(dist, Leg):
    """Take mean of distribution 'dist' to find average X = E[x]"""
    p, qf, xstar, x_l, x_h = dist
    wleg, xleg = leg.interval(Leg, x_l, x_h)
    return p * xstar + wleg @ (cheb.val_array(qf, xleg)*xleg)


def steady_state(c, ideal, beta, lamb, mu, sigma, a, b, n_cheb=25, n_leg=25, backward_tol=1E-13, forward_tol=1E-9, verbose=True):
    """Convenience function that calculates steady-state policy (xstar, x_l, x_h), beginning-of-period
    value function W(x), and distribution (p, f), given parameters of the steady-state problem, an interval [a, b]
    on which to obtain W(x) (ideally tight around x_l, x_h), and parameters for numerical solution."""
    # make sure all arguments are floats so that Numba doesn't need to recompile a bunch of times (also this caused some error)
    c, ideal, beta, lamb, mu, sigma, a, b = (float(x) for x in (c, ideal, beta, lamb, mu, sigma, a, b))

    # precalculate Chebyshev nodes and interpolation matrix and Legendre nodes and weights, all on [-1, 1]
    Cheb = cheb.start(n_cheb)
    Leg = leg.start(n_leg)

    # get steady-state policies (xstar, x_l, x_h) and beginning-of-period value function 'W' on [a, b]
    qW, xstar, x_l, x_h = steady_state_policy(c, ideal, beta, lamb, mu, sigma, a, b, Cheb, Leg, backward_tol, verbose)

    # use these to get steady-state mass 'p' at reset point xstar, and density 'f' on interval [x_l, x_h],
    dist = steady_state_dist(xstar, x_l, x_h, lamb, mu, sigma, Cheb, Leg, forward_tol, verbose)
    p, qf = dist[0], dist[1]

    # take mean of distribution to find average X = E[x]
    X = aggregate(dist, Leg)

    return qW, xstar, x_l, x_h, p, qf, X


"""Get E^s(x) functions for Jacobian"""

@njit
def expectations_iterate(qE_prime, policy, params):
    """Take expectations iteraction of any end-of-period function going backward, from E_prime to E,
    on the interval [x_l, x_h], assuming constant steady-state policy"""
    # unpack steady-state policy and constant parameters
    xstar, x_l, x_h = policy
    la, mu, sigma, Cheb, Leg = params
    
    # what is E_prime at xstar?
    E_prime_xstar = cheb.val_scalar(qE_prime, xstar)
    
    # specialize Chebyshev to this interval (not necessary since constant)
    P, x = cheb.interval(Cheb, x_l, x_h)
    
    # at each Chebyshev node x in [x_l, x_h], take expectations of E_prime
    f = np.empty_like(x)

    for i in range(len(x)):
        xi = x[i]
        exp_reset =  ((1-la) * (1 - probability_within_interval(x_l, x_h, xi + mu, sigma)) + la) * E_prime_xstar
        exp_noreset = (1-la) * integrate_within_interval(qE_prime, x_l, x_h, xi + mu, sigma, Leg)
        f[i] = exp_reset + exp_noreset

    return cheb.interp(P, f)


def get_Es(policy, lamb, mu, sigma, T, n_cheb=25, n_leg=25):
    Cheb = cheb.start(n_cheb)
    Leg = leg.start(n_leg)
    return get_Es_inner(policy, lamb, mu, sigma, T, Cheb, Leg)


@njit
def get_Es_inner(policy, lamb, mu, sigma, T, Cheb, Leg):
    """Given policy (xstar, x_l, x_h) and innovations N(mu, sigma^2), return functions E^s(x)
    on [x_l, x_h] for all s = 0,...,T-1. Return Chebyshev coefficients directly in T*n_cheb array."""
    # prep work: create params object
    params = (lamb, mu, sigma, Cheb, Leg)
    
    xstar, x_l, x_h = policy
    P, x = cheb.interval(Cheb, x_l, x_h)
    Es = np.empty((T, len(x)))

    # E^0(x) = x
    Es[0] = cheb.interp(P, x)[0]
    
    for t in range(1, T):
        Es[t] = expectations_iterate((Es[t-1], x_l, x_h), policy, params)[0]
        
    return Es


def get_equivalent_td(policy, dist, Es):
    """Given Es, policy, and distribution find the price survival functions Phi for
    the equivalent time-dependent pricers corresponding to the intensive and 2 extensive
    margins, in addition to their weights."""
    xstar, x_l, x_h = policy
    p, qf, *_ = dist
    
    T = len(Es)
    Phis = np.empty((T, 3))

    # many ways to make this more efficient, but not really necessary, so doing this for clarity
    Phis[:, 0] = np.array([cheb.val_scalar(cheb.der((Es[t], x_l, x_h)), xstar) for t in range(T)])

    Es_xstar = np.array([cheb.val_scalar((Es[t], x_l, x_h), xstar) for t in range(T)])
    Phis[:, 1] = - (np.array([cheb.val_scalar((Es[t], x_l, x_h), x_l) for t in range(T)]) - Es_xstar)
    Phis[:, 2] = np.array([cheb.val_scalar((Es[t], x_l, x_h), x_h) for t in range(T)]) - Es_xstar

    # weights should be sum of Es times density / mass, correct for any slight failure to sum to 1
    weights = Phis.sum(axis=0) * np.array([p, cheb.val(qf, x_l), cheb.val(qf, x_h)])
    weights /= weights.sum()

    return weights, Phis


"""Support functions for iterations"""


@njit
def get_optimal_reset_and_thresholds(qV, c):
    """Given end-of-period value function V on [a, b] and cost of adjustment c, solve FOCs
    to get reset policy xstar and Ss adjustment thresholds [x_l, x_h]"""
    # extract interval [a, b], precalculate derivatives
    a, b = qV[1], qV[2]
    qV_p = cheb.der(qV)
    qV_pp = cheb.der(qV_p)
    
    # run newton's method solving FOC (3), with seed in middle of interval, to get xstar
    xstar = cheb.newton(qV_p, qV_pp, 0, (a+b)/2)
    Vxstar = cheb.val_scalar(qV, xstar)
    
    # run newton's method with a, b as guesses to get x_l and x_h using FOCs (1) and (2)
    x_l = cheb.newton(qV, qV_p, Vxstar + c, a)
    x_h = cheb.newton(qV, qV_p, Vxstar + c, b)
    
    return xstar, x_l, x_h, Vxstar


@njit
def p_next(p_lag, qf_lag, xstar_lag, x_l_lag, x_h_lag, x_l, x_h, lamb, mu, sigma, Leg):
    """Given yesterday's end-of-period distribution, both mass point p_lag at xstar_lag and density
    f_lag over [x_l_lag, x_h_lag], what is today's mass point p after N(mu, sigma^2) innovations, free resets,
    and endogenous resets?"""
    # mass point today coming from mass point yesterday
    from_mass = p_lag * (1 - probability_within_interval(x_l, x_h, mu + xstar_lag, sigma))
    
    # mass point today coming from density yesterday
    # (integrate probability going to mass point against yesterday's density f_lag over [x_l_lag, x_h_lag])
    from_density = 0
    wleg_lag, xleg_lag = leg.interval(Leg, x_l_lag, x_h_lag)
    for i in range(len(xleg_lag)):
        from_density += (wleg_lag[i] * (1 - probability_within_interval(x_l, x_h, mu + xleg_lag[i], sigma))
                                                                        * cheb.val_scalar(qf_lag, xleg_lag[i]))
    
    return (1-lamb) * (from_mass + from_density) + lamb


@njit
def f_next(p_lag, qf_lag, xstar_lag, x_l_lag, x_h_lag, x_l, x_h, lamb, mu, sigma, Cheb, Leg):    
    """Given yesterday's end-of-period distribution, both mass point p_lag at xstar_lag and density
    f_lag over [x_l_lag, x_h_lag], what is today's density f over [x_l, x_h] after N(mu, sigma^2)
    innovations and free resets?"""

    # remap Chebyshev nodes to today's [x_l, x_h]
    P, x = cheb.interval(Cheb, x_l, x_h)
    f = np.empty_like(x)
    for i in range(len(x)):
        xi = x[i]
        # then at each node, find density of getting there from mass point
        # and integrate probability of transition against yesterday's density to find rest of today's density
        f[i] = p_lag*normal_pdf(xi-xstar_lag, mu, sigma) + integrate_within_interval(qf_lag, x_l_lag, x_h_lag, xi-mu, sigma, Leg)
    
    return cheb.interp(P, (1-lamb)*f)


"""Math helper functions"""


@njit
def integrate_within_interval(qF, x_l, x_h, mu, sigma, Leg):
    """Take expectation of F(x)*1(x in [x_l, x_h]) if x is distributed N(mu, sigma^2)"""
    aquad = max(mu - 8*sigma, x_l)
    bquad = min(mu + 8*sigma, x_h)
    w, x = leg.interval(Leg, aquad, bquad)
    return w @ (normal_pdf(x, mu, sigma)*cheb.val_array(qF, x))


@njit
def probability_within_interval(x_l, x_h, mu, sigma):
    """Probability that x lies in interval [x_l, x_h] if distribution N(mu, sigma^2)"""
    return normal_cdf(x_h, mu, sigma) - normal_cdf(x_l, mu, sigma)


@njit
def normal_pdf(x, mu, sigma):
    return np.exp(-((x-mu)/sigma)**2/2)/np.sqrt(2*np.pi)/sigma


@njit
def normal_cdf(x, mu, sigma):
    z = (x - mu)/sigma
    return (1 + math.erf(z / np.sqrt(2))) / 2


