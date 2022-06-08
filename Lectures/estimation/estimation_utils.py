"""Useful functions"""

import copy
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.signal import detrend
from numba import njit, prange
from pandas_datareader.fred import FredReader

from sequence_jacobian import estimation as ssj_est

np.random.seed(1234)


''' IRF '''


def irf_make(G, shock, shockname, varlist, T):
    irf = {}
    for i, var in enumerate(varlist): 
        if var == shockname:
            irf[var] = shock
        else: 
            try:
                irf[var] = G[var][shockname] @ shock
            except:
                irf[var] = np.zeros(T)
    return irf


''' Simulation '''


def simulate(G, inputs, outputs, rhos, sigmas, T, T_simul):
    '''
    G: model jacobian
    inputs: e.g. ['ishock','X']
    outputs: e.g. ['pi','Y']
    rhos: dictionary with shocks persistence e.g. {'ishock': 0.9, 'X': 0.5}
    sig: dictionary with shocks standard deviations e.g. {'ishock': 1, 'X': 1}
    T: size of G
    T_sim: length of simulated periods
    '''
    epsilons = {i: np.random.randn(T_simul + T - 1) for i in inputs}
    simulations = {}
    
    for o in outputs:
        dXs = {i: sigmas[i]*(G[o][i] @ rhos[i] ** np.arange(T)) for i in inputs}
        simulations[o] = sum(simul_shock(dXs[i], epsilons[i]) for i in inputs)
    
    return simulations


@njit(parallel=True)
def simul_shock(dX, epsilon):
    dX_flipped = dX[::-1].copy() # flip so we don't need to flip epsilon
    T = len(dX)
    T_simul = len(epsilon)
    Y = np.empty(T_simul - T + 1)
    for t in prange(T_simul - T + 1):
        Y[t] = np.vdot(dX_flipped, epsilon[t:t+T])
    return Y


''' Estimation '''


def loglik_f(x, data, outputs, shock_series, priors_list, T, G): 
    
    '''
    x: array with shock process parameters e.g. (sd1, rho1, sd2, rho2, etc.)
    data: np.array(T,n_output), same order as x
    outputs: series in the data e.g. ['pi']
    shock_series: list of AR(k) shocks e.g. [('ishock', 1), ('X', 1)] for AR(1) processes
    priors_list: list of priors, same order as x e.g. [('Uniform',0,2),('Invgamma',0,2)]
    T: size of G
    G: model jacobian
    '''
    
    T_irf = T - 20
    n_se,n_sh = len(outputs), len(shock_series)
    meas_error = np.zeros(n_se)                  # set measurement error to zero

    # extract shock parameters from x; order: always sigma first, then AR coefs, then MA coefs
    sigmas, arcoefs, macoefs = get_shocks_arma(x, shock_series)

    # Step 1
    As = step1_est(G, arcoefs, macoefs, shock_series, outputs, T, T_irf, n_se, n_sh)

    # Step 2
    Sigma = ssj_est.all_covariances(As, sigmas)

    # Step 3
    llh = step3_est(Sigma, data, sigma_o=meas_error)

    # compute the posterior by adding the log prior
    log_posterior = llh + log_priors(x, priors_list)
    
    return log_posterior


def get_shocks_arma(x, shock_series):
    ix, ishock = 0, 0
    sigmas, arcoefs, macoefs = np.zeros((3, len(shock_series)))
    for shock_name, order in shock_series:
        sigmas[ishock] = x[ix]
        ix += 1
        if order >= 1:
            arcoefs[ishock] = x[ix]
            ix += 1
        if order >= 2:
            macoefs[ishock] = x[ix]
            ix += 1
        ishock += 1
    return sigmas, arcoefs, macoefs


def step1_est(G, arcoefs, macoefs, shock_series, outputs, T, T_irf, n_se, n_sh):
    ''' Compute the MA representation given G '''
    # Compute MA representation of outcomes in As
    As = np.empty((T_irf, n_se, n_sh))
    for i_sh in range(n_sh):
        arma_shock = arma_irf(np.array([arcoefs[i_sh]]), np.array([macoefs[i_sh]]), T)

        if np.abs(arma_shock[-1]) > 1e20:
            raise Warning('ARMA shock misspecified, leading to explosive shock path!')

        # store for each series
        shockname = shock_series[i_sh][0]
        for i_se in range(n_se):
            As[:, i_se, i_sh] = (G[outputs[i_se]][shockname] @ arma_shock)[:T_irf]
    return As


@njit
def arma_irf(ar_coeff, ma_coeff, T):  # NOTE: these need to be numpy.ndarrays
    """ Generates shock IRF for any ARMA process """
    x = np.empty((T,))
    n_ar = ar_coeff.size
    n_ma = ma_coeff.size
    sign_ma = 1                             # in SW all MA coefficients are multiplied by -1
    for t in range(T):
        if t == 0:
            x[t] = 1
        else:
            ar_sum = 0
            for i in range(min(n_ar, t)):
                ar_sum += ar_coeff[i] * x[t-1-i]
            ma_term = 0
            if 0 < t <= n_ma:
                ma_term = ma_coeff[t-1]
            x[t] = ar_sum + ma_term * sign_ma
    return x


def step3_est(Sigma, y, sigma_o=None):
    To, O = y.shape
    loglik = (ssj_est.log_likelihood(y, Sigma, sigma_o) - (To * O * np.log(2 * np.pi))/2)
    return loglik


def log_priors(x, priors_list):
    """This function computes a sum of log prior distributions that are stored in priors_list.
    Example: priors_list = {('Normal', 0, 1), ('Invgamma', 1, 2)}
    and x = np.array([1, 2])"""
    assert len(x) == len(priors_list)
    sum_log_priors = 0
    for n in range(len(x)):
        dist = priors_list[n][0]
        mu = priors_list[n][1]
        sig = priors_list[n][2]
        if dist == 'Normal':
            sum_log_priors += - 0.5 * ((x[n] - mu)/sig)**2
        elif dist == 'Uniform':
            lb = mu
            ub = sig
            sum_log_priors += - np.log(ub-lb)
        elif dist == 'Invgamma':
            s = mu
            v = sig
            sum_log_priors += (-v-1) * np.log(x[n]) - v*s**2/(2*x[n]**2)
        elif dist == 'Gamma':
            theta = sig**2 / mu
            k = mu / theta
            sum_log_priors += (k-1) * np.log(x[n]) - x[n]/theta
        elif dist == 'Beta':
            alpha = (mu*(1 - mu) - sig**2) / (sig**2 / mu)
            beta = alpha / mu - alpha
            sum_log_priors += (alpha-1) * np.log(x[n]) + (beta-1) * np.log(1-x[n])
        else:
            raise ValueError('Distribution provided is not implemented in log_priors!')

    if np.isinf(sum_log_priors) or np.isnan(sum_log_priors):
        print(x)
        raise ValueError('Need tighter bounds to prevent prior value = 0')
    return sum_log_priors


''' Historical decomposition '''


def back_out_shocks(As, y, sigma_e=None, sigma_o=None, preperiods=0):
    """Calculates most likely shock paths if As is true set of IRFs

    Parameters
    ----------
    As : array (Tm*O*E) giving the O*E matrix mapping shocks to observables at each of Tm lags in the MA(infty),
            e.g. As[6, 3, 5] gives the impact of shock 5, 6 periods ago, on observable 3 today
    y : array (To*O) giving the data (already assumed to be demeaned, though no correction is made for this in the log-likelihood)
            each of the To rows t is the vector of observables at date t (earliest should be listed first)
    sigma_e : [optional] array (E) giving sd of each shock e, assumed to be 1 if not provided
    sigma_o : [optional] array (O) giving sd of iid measurement error for each observable o, assumed to be 0 if not provided
    preperiods : [optional] integer number of pre-periods during which we allow for shocks too. This is suggested to be at
            least 1 in models where some variables (e.g. investment) only respond with a 1 period lag.
            (Otherwise there can be invertibility issues)

    Returns
    ----------
    eps_hat : array (To*E) giving most likely path of all shocks
    Ds : array (To*O*E) giving the level of each observed data series that is accounted for by each shock
    """
    # Step 1: Rescale As any y
    To, Oy = y.shape
    Tm, O, E = As.shape
    assert Oy == O
    To_with_pre = To + preperiods

    A_full = construct_stacked_A(As, To=To_with_pre, To_out=To, sigma_e=sigma_e, sigma_o=sigma_o)
    if sigma_o is not None:
        y = y / sigma_o
    y = y.reshape(To*O)

    # Step 2: Solve OLS
    eps_hat = np.linalg.lstsq(A_full, y, rcond=None)[0]  # this is To*E x 1 dimensional array
    eps_hat = eps_hat.reshape((To_with_pre, E))

    # Step 3: Decompose data
    for e in range(E):
        A_full = A_full.reshape((To,O,To_with_pre,E))
        Ds = np.sum(A_full * eps_hat,axis=2)

    # Cut away pre periods from eps_hat
    eps_hat = eps_hat[preperiods:, :]

    return eps_hat, Ds


def construct_stacked_A(As, To, To_out=None, sigma_e=None, sigma_o=None, reshape=True, long=False):
    Tm, O, E = As.shape

    # how long should the IRFs be that we stack in A_full?
    if To_out is None:
        To_out = To
    if long:
        To_out = To + Tm  # store even the last shock's IRF in full!

    # allocate memory for A_full
    A_full = np.zeros((To_out, O, To, E))

    for o in range(O):
        for itshock in range(To):
            # if To > To_out, allow the first To - To_out shocks to happen before the To_out time periods
            if To <= To_out:
                iA_full = itshock
                iAs = 0

                shock_length = min(Tm, To_out - iA_full)
            else:
                # this would be the correct start time of the shock
                iA_full = itshock - (To - To_out)

                # since it can be negative, only start IRFs at later date
                iAs = - min(iA_full, 0)

                # correct iA_full by that date
                iA_full += - min(iA_full, 0)

                shock_length = min(Tm, To_out - iA_full)

            for e in range(E):
                A_full[iA_full:iA_full + shock_length, o, itshock, e] = As[iAs:iAs + shock_length, o, e]
                if sigma_e is not None:
                    A_full[iA_full:iA_full + shock_length, o, itshock, e] *= sigma_e[e]
                if sigma_o is not None:
                    A_full[iA_full:iA_full + shock_length, o, itshock, e] /= sigma_o[o]
    if reshape:
        A_full = A_full.reshape((To_out * O, To * E))
    return A_full


def estimate(outputs, data, x_guess, shock_series, priors_list, T, G=None, params=None, jac_info=None, sd=True,
             data_demean_f=False, **kwargs):
    if G is None and jac_info is None:
        raise ValueError('Need at least G or jac_info and params as input!')

    # If we do not need to compute the model jacobian G
    if G is not None:
        def objective(x):
            return -loglik_f(x, data, outputs, shock_series, priors_list, T, G)
    # If we do
    else:
        # Store model jacobian when necessary
        n_params = len(params)
        last_x_params = [np.zeros(n_params)]
        last_G = [{}]
        if 'Js' not in jac_info:
            jac_info['Js'] = {}

        def objective(x):
            # check whether we estimate the intercept of the data
            if data_demean_f == False:
                data_adj = data.copy()
            else:
                data_adj = data_demean_f(x,data)

            # Update parameters
            x_params = x[-n_params:]

            # Check whether params have changed or not since last iteration
            if not np.allclose(x_params, last_x_params[0], rtol=1e-12, atol=1e-12):
                print('could not reuse')
                # write new parameters into ss
                ss = copy.deepcopy(jac_info['ss'])
                ss.update({param: x_params[j] for j, param in enumerate(params)})

                # Compute model jacobian G
                G = jac_info['model'].solve_jacobian(ss, unknowns=jac_info['unknowns'], targets=jac_info['targets'],
                                                     inputs=jac_info['exogenous'], Js=jac_info['Js'], T=T)

                # Store for later
                last_x_params[0] = x_params.copy()
                last_G[0] = G
            else:
                print('got to reuse')
                # if not, re-use the one from before
                G = last_G[0]

            # Compute log likelihood
            return -loglik_f(x, data_adj, outputs, shock_series, priors_list, T, G)

    # minimize objective
    result = opt.minimize(objective, x_guess, **kwargs)

    # Compute standard deviation if required
    if sd:
        H, nfev_total = hessian(objective, result.x, nfev=result.nfev, f_x0=result.fun)
        Hinv = np.linalg.inv(H)
        x_sd = np.sqrt(np.diagonal(Hinv))
    else:
        nfev_total = result.nfev
        x_sd = np.zeros_like(result.x)

    return result, x_sd, nfev_total


def hessian(f, x0, nfev=0, f_x0=None, dx=1e-4):
    """Function to compute Hessian of generic function"""
    n = x0.shape[0]
    I = np.eye(n)

    # check if function value is given
    if f_x0 is None:
        f_x0 = f(x0)
        nfev += 1

    # compute Jacobian
    jac = np.empty(n)
    for i in range(n):
        jac[i] = (f_x0 - f(x0 - dx * I[i,:])) / dx
        nfev += 1

    # compute the hessian)
    hess = np.empty((n,n))
    for i in range(n):
        f_xi = f(x0 + dx * I[i, :])
        nfev += 1
        hess[i, i] = ((f_xi - f_x0) / dx - jac[i]) / dx
        for j in range(i):
            jac_j_at_xi = (f(x0 + dx * I[i,:] + dx * I[j,:]) - f_xi) / dx
            nfev += 1
            hess[i, j] = (jac_j_at_xi - jac[j] ) / dx - hess[j,j]
            hess[j, i] = hess[i, j]

    return hess, nfev


def construct_us_data(load_path='', start='1966-01', end='2004-12'):
    if load_path != '':
        df = pd.read_csv(load_path)
    else:
        # Define series to load from Fred
        series = {"GDPDEF": "gdpdef",             # GDP deflator (index 2012) [Q]
                  "PCEPILFE": "pcecore",          # PCE deflator excl food and energy (index 2012) [M]
                  "GDP": "ngdp",                  # Nominal GDP [Q]
                  "FEDFUNDS": "ffr",              # Fed Funds rate
                  "USREC": "USREC"}               # NBER US recession indicator

        # Load series from Fred
        df_fred = FredReader(series.keys(), start='1959-01').read().rename(series, axis='columns')
        df_fred.index = df_fred.index.to_period('M')

        # make everything quarterly
        df_fred = df_fred.groupby(pd.PeriodIndex(df_fred.index, freq='Q')).mean()
        df_fred["USREC"] = round(df_fred["USREC"])

        # start new dataframe into which we'll selectively load
        df = pd.DataFrame(index=df_fred.index)
        df['USREC'] = df_fred["USREC"]

        # Load series
        df['pi'] = 100*((df_fred['pcecore']/df_fred['pcecore'].shift(1))**4-1)  # inflation is PCE
        df[['Y']] = df_fred[['ngdp']].div(df_fred['gdpdef'], axis=0)   # gdp = nominal gdp / gdp deflator
        df['i'] = df_fred['ffr']  # fed funds rate

        # only keep start to end and define time variable
        df = df[start:end]
        df.index.name = 't'

        # Detrend or demean
        for k in ('Y'):
            df[k] = 100 * detrend(np.log(df[k]))  # for non-rate variables, take logs, remove trend, multiply by 100
        for k in ('pi', 'i'):
            df[k] = df[k] - df[k].mean()  # inflation and interest rates just remove constant

        # Create lags and leads
        df['Y_plus1'] = df['Y'].shift(1)
        df['pi_plus1'] = df['pi'].shift(1)
        df['pi_minus1'] = df['pi'].shift(-1)
        df['i_plus1'] = df['i'].shift(1)
        df['i_minus1'] = df['i'].shift(-1)

    # get data in the array format for estimation (3, T_data)
    data = df[['pi', 'Y', 'i']].values       # Make sure the order here is the same as in outputs!

    return data, df

