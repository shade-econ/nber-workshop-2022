"""Model blocks and helper functions for the estimation tutorial"""

import numpy as np
import matplotlib.pyplot as plt

import sequence_jacobian as sj
from estimation import estimation_utils as est_util
# import estimation_utils as est_util


def income_incidence(e_grid, Y, T):
    y = (Y-T) * e_grid
    return y


def make_grids(min_a, max_a, n_a, rho_e, sd_e, n_e):
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    e_grid, e_pdf, Pi = sj.grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    return a_grid, e_grid, e_pdf, Pi


def household_init(rpost, Y, T, eis, a_grid, e_grid):
    # initialize guess for policy function iteration
    coh = (1 + rpost) * a_grid[np.newaxis, :] + (Y-T) * e_grid[:, np.newaxis]
    Va = (1 + rpost) * (0.2 * coh) ** (-1/eis)
    return Va


# hh = sj.hetblocks.hh_sim.hh.add_hetinputs([make_grids, income_incidence])
@sj.het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, min_a, e_grid, y, rpost, beta, eis):
    """Single backward iteration step using endogenous gridpoint method for households with separable utility.
    y_grid is assumed to be an (nS) array, r is ex-post interest rate in period t
    """
    # backward step
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    lhs = c_nextgrid + a_grid[np.newaxis, :] - y[:, np.newaxis]
    rhs = (1 + rpost) * a_grid
    a = sj.interpolate.interpolate_y(lhs, rhs, a_grid)
    sj.misc.setmin(a, min_a)
    c = rhs[np.newaxis, :] + y[:, np.newaxis] - a
    uc = c ** (-1/eis)
    Va = (1 + rpost) * uc

    # other outputs (will need for NKPC)
    uce = uc * e_grid[:, np.newaxis]

    return Va, a, c, uc, uce


hh = household.add_hetinputs([make_grids, income_incidence])


@sj.simple
def rpost_simple(r):
    rpost = r(-1)
    return rpost


@sj.solved(unknowns={'B': (-1., 1.)}, targets=['Bres'], solver="brentq")
def fiscal_deficit_Trule(r, G, B, Tss, phi_T):
    T = Tss + phi_T * (B(-1) - B.ss)
    Bres = (1 + r(-1)) * B(-1) + G - T - B
    return T, Bres


@sj.simple
def nkpc(pi, Y, X, C, kappa_w, vphi, frisch, markup_ss, eis, beta):
    piw = pi + X - X(-1)
    piwres = kappa_w * (vphi*(Y/X)**(1/frisch) - 1/markup_ss * X * C**-(1/eis)) + beta * piw(1) - piw
    return piwres, piw


@sj.simple
def monetary_taylor(pi, ishock, rss, phi_pi):
    i = rss + phi_pi * pi + ishock
    r = i - pi(1)
    return i, r


@sj.simple
def mkt_clearing(A, B, Y, C, G):
    asset_mkt = A - B
    goods_mkt = C + G - Y
    return asset_mkt, goods_mkt


ha = sj.create_model([hh, rpost_simple, fiscal_deficit_Trule, nkpc, monetary_taylor, mkt_clearing],
                     name="Simple HA Model")


def setup_us_data_est(ss):
    outputs = ['pi', 'Y', 'i']
    params = ['kappa_w', 'phi_pi', 'phi_T']  # define here the model parameters that we estimate
    shock_series = [('ishock', 1), ('X', 1), ('G', 1)]
    priors = [('Invgamma', 0.4, 4), ('Uniform', 0, 1), ('Invgamma', 0.5, 4), ('Uniform', 0, 1), ('Invgamma', 1, 4),
              ('Uniform', 0, 1),
              ('Uniform', 0, 1), ('Gamma', 1.5, 0.25), ('Uniform', 0, 1)]  # shock parameters then model parameters
    bounds = [(0.05, 10), (0.01, 0.99)] * len(shock_series) + [(0.01, 0.99), (1.05, 5),
                                                               (0.01, 1)]  # shock parameters then model parameters
    x_guess = [1, 0.5] * len(shock_series) + [ss[k] for k in params]  # shock parameters then model parameters
    return outputs, params, shock_series, priors, bounds, x_guess


# Figure generating code
def figure_simul(data_simul, Tplot=1001):
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1)
    plt.plot(data_simul['pi'][:Tplot])
    plt.title(r'$\pi$')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.xlabel('Quarters')
    plt.subplot(1,3,2)
    plt.plot(data_simul['Y'][:Tplot])
    plt.title(r'$Y$')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.xlabel('Quarters')
    plt.subplot(1,3,3)
    plt.plot(data_simul['i'][:Tplot])
    plt.title(r'$i$')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.xlabel('Quarters')
    plt.tight_layout()
    plt.show()


def figure_us_data(df):
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1)
    df['pi'].plot()
    yl, yh = plt.ylim()
    plt.fill_between(df.index, yl, yh, where=df['USREC'].values, color='k', alpha=0.1)
    plt.title(r'$\pi$')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.xlabel('Quarters')
    plt.subplot(1,3,2)
    df['Y'].plot()
    yl, yh = plt.ylim()
    plt.fill_between(df.index, yl, yh, where=df['USREC'].values, color='k', alpha=0.1)
    plt.title(r'$Y$')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.xlabel('Quarters')
    plt.subplot(1,3,3)
    df['i'].plot()
    yl, yh = plt.ylim()
    plt.fill_between(df.index, yl, yh, where=df['USREC'].values, color='k', alpha=0.1)
    plt.title(r'$i$')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.xlabel('Quarters')
    plt.tight_layout()
    plt.show()


def figure_mode_irf(x_ha2, G_ha, shock_list, varlist, T):
    # HA model
    irf_ha = {}
    shock = {}
    shock['ishock'] = x_ha2[0] * x_ha2[1] **np.arange(T)
    shock['X'] = x_ha2[2] * x_ha2[3] ** np.arange(T)
    shock['G'] = x_ha2[4] * x_ha2[5] ** np.arange(T)
    for s in shock_list:
        irf_ha[s] = est_util.irf_make(G_ha, shock[s], s, varlist, T)

    # plot
    Tplot = 30
    plt.figure(figsize=(12.5, 12))
    for i,k in enumerate(varlist):
        plt.subplot(3,3,i+1)
        plt.title(k)
        plt.plot(irf_ha['ishock'][k][0:Tplot], color='tab:blue', label = 'Monetary shock')
        plt.plot(irf_ha['X'][k][0:Tplot], color='tab:orange', label = 'Productivity shock')
        plt.plot(irf_ha['G'][k][0:Tplot], color='tab:red', label = 'Government spending shock')
        plt.axhline(y=0, color='#808080', linestyle=':')
        plt.xlabel('Quarters')
        if k =='C':
            plt.legend(framealpha=0)
    plt.tight_layout()
    plt.show()


def figure_hist_decomp(x, G, shock_series, outputs, data, T, T_sim, df=None):
    # extract shock parameters from x
    sigmas, arcoefs, macoefs = est_util.get_shocks_arma(x, shock_series)
    As = est_util.step1_est(G, arcoefs, macoefs, shock_series, outputs, T, T - 20, len(outputs), len(shock_series))

    # Compute shock decomposition
    eps_hat_est, Ds = est_util.back_out_shocks(As, data, sigma_e=sigmas, sigma_o=None, preperiods=1)  # allow for 1 pre-period with inv 1 period in advance
    if df is None:
        xaxis = np.arange(T_sim)
    else:
        xaxis = df.index[0].year + 0.25*np.arange(4*(df.index[-1].year-df.index[0].year+1))

    # plot
    Tshow, len_se, len_sh = Ds.shape
    colors = {'ishock': 'C0','X': 'C1','G': 'C2'}
    plt.figure(figsize=(12, 4))
    for i,s in enumerate(outputs):
        plt.subplot(1,3,1+i)
        series_fig = s
        i_se = outputs.index(series_fig)
        y_offset_pos = np.zeros_like(Ds[:,i_se,0])  # positive offset, collecting positive terms
        y_offset_neg = np.zeros_like(Ds[:, i_se, 0])  # negative offset, collecting negative terms

        # loop over shocks
        for i_sh in range(len_sh):

            sh = shock_series[i_sh][0]
            Ds_here = Ds[:,i_se,i_sh]
            y_offset = (Ds_here > 0) * y_offset_pos + (Ds_here < 0) * y_offset_neg
            y_offset_pos_ = y_offset_pos + np.maximum(Ds_here,0)
            y_offset_neg_ = y_offset_neg - np.maximum(-Ds_here,0)
            plt.fill_between(xaxis, y_offset_pos, y_offset_pos_, color=colors[sh], label=sh)
            plt.fill_between(xaxis, y_offset_neg, y_offset_neg_, color=colors[sh])
            y_offset_pos = y_offset_pos_
            y_offset_neg = y_offset_neg_

        plt.plot(xaxis,data[:,i_se],color='black')
        if i == 0: plt.legend(framealpha=1)
        plt.title(series_fig);

    plt.tight_layout();
    plt.show()


def figure_fevd(x_ha2, G, shock_series, outputs, T):
    # extract shock parameters from x
    sigmas, arcoefs, macoefs = est_util.get_shocks_arma(x_ha2, shock_series)
    As = est_util.step1_est(G, arcoefs, macoefs, shock_series, outputs, T, T - 20, len(outputs), len(shock_series))

    # Compute forecast error
    FEV = np.cumsum((As * sigmas) ** 2, axis=0)     # compute forecast error variance contributions for every shock and every series over time
    FEV_sum = np.sum(FEV, axis=2)
    FEV_sum[FEV_sum == 0] = np.inf
    FEV_invsum_by_t_series_ra = 1 / FEV_sum         # normalize
    FEV_norm = FEV_invsum_by_t_series_ra[..., np.newaxis] * FEV

    # plot
    horizon = 100
    xaxis = np.arange(horizon)
    Ds = FEV_norm[:horizon, ...]
    Tshow, len_se, len_sh = Ds.shape
    colors = {'ishock': 'C0','X': 'C1','G': 'C2'}
    plt.figure(figsize=(12, 4))
    for i,s in enumerate(outputs):
        plt.subplot(1,3,1+i)
        series_fig = s
        i_se = outputs.index(series_fig)
        y_offset_pos = np.zeros_like(Ds[:,i_se,0])  # positive offset, collecting positive terms
        y_offset_neg = np.zeros_like(Ds[:, i_se, 0])  # negative offset, collecting negative terms

        # loop over shocks
        for i_sh in range(len_sh):

            sh = shock_series[i_sh][0]
            Ds_here = Ds[:,i_se,i_sh]
            y_offset = (Ds_here > 0) * y_offset_pos + (Ds_here < 0) * y_offset_neg
            y_offset_pos_ = y_offset_pos + np.maximum(Ds_here,0)
            y_offset_neg_ = y_offset_neg - np.maximum(-Ds_here,0)
            plt.fill_between(xaxis, y_offset_pos, y_offset_pos_, color=colors[sh], label=sh)
            plt.fill_between(xaxis, y_offset_neg, y_offset_neg_, color=colors[sh])
            y_offset_pos = y_offset_pos_
            y_offset_neg = y_offset_neg_

        if i == 0: plt.legend(framealpha=1)
        plt.title(series_fig);

    plt.tight_layout()
    plt.show()

