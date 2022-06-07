"""Helper and plotting functions for the closed economy monetary tutorial"""

import numpy as np
import matplotlib.pyplot as plt
import sequence_jacobian as sj


def compute_ra_impcs(beta, T=300):
    impcs = (1 - beta) * beta ** (np.tile(np.arange(T), (T, 1)))
    return impcs

# Plotting code
def figure_1(irf, ss, dr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
    ax1.set_title('Output response')
    ax1.plot(100 * irf['ha']['Y'][0:20] / ss['ha']['Y'], label='HA')
    ax1.plot(100 * irf['ra']['Y'][:20] / ss['ra']['Y'], linestyle = '--', label='RA')
    ax1.axhline(y=0, color='#808080', linestyle=':')
    ax1.legend(framealpha=0)
    ax1.set_ylabel('% deviation from ss')
    ax1.set_xlabel(r"Year $(t)$")
    ax2.set_title('Real rate shock')
    ax2.plot(100 * dr[0:20], label='r')
    ax2.axhline(y=0, color='#808080', linestyle=':')
    ax2.legend(framealpha=0)
    ax2.set_xlabel(r"Year $(t)$")
    ax2.set_ylabel('pp deviation from ss')
    plt.tight_layout()
    #plt.savefig(f'Export/HAvsRA_lec2.pdf', format='pdf', transparent=True);
    plt.show()

    # Plot
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(8, 3.8))
    ax3.set_title('Output response')
    ax3.plot(100 * irf['ha_zl']['Y'][0:20] / ss['ha_zl']['Y'], label='HA w/ Zero Liq.')
    ax3.plot(100 * irf['ra']['Y'][:20] / ss['ha_zl']['Y'], linestyle = '--', label='RA')
    ax3.axhline(y=0, color='#808080', linestyle=':')
    ax3.legend(framealpha=0)
    ax3.set_ylabel('% deviation from ss')
    ax3.set_xlabel(r"Year $(t)$")
    ax4.set_title('Real rate shock')
    ax4.plot(100 * dr[0:20], label='r')
    ax4.axhline(y=0, color='#808080', linestyle=':')
    ax4.legend(framealpha=0)
    ax4.set_xlabel(r"Year $(t)$")
    ax4.set_ylabel('pp deviation from ss')
    plt.tight_layout()
    #plt.savefig(f'Export/HAvsRA_zeroliq_lec2.pdf', format='pdf', transparent=True);
    plt.show()


def figure_2(dC, dC_dr, dC_dY):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
    ax1.set_title('HA model')
    ax1.plot(100 * dC['ha'][:20], label='total', color='black')
    ax1.plot(100 * dC_dr['ha'][:20], label='direct')
    ax1.plot(100 * dC_dY['ha'][:20], label='indirect')
    ax1.axhline(y=0, color='#808080', linestyle=':')
    ax1.set_xlabel(r"Year $(t)$")
    ax1.set_ylabel('% deviation from ss')
    ax1.set_ylim([-0.25,2])
    ax1.legend(framealpha=0)
    ax2.set_title('RA model')
    ax2.plot(100 * dC['ra'][:20], label='total', color='black')
    ax2.plot(100 * dC_dr['ra'][:20], label='direct')
    ax2.plot(100 * dC_dY['ra'][:20], label='indirect')
    ax2.axhline(y=0, color='#808080', linestyle=':')
    ax2.set_xlabel(r"Year $(t)$")
    ax2.set_ylim([-0.25,2])
    ax2.legend(framealpha=0)
    plt.tight_layout()
    #plt.savefig('Export/directindirect_lec2.pdf', format='pdf', transparent=True)
    plt.show()


def figure_3(G, Tshock=10):
    fig = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot(-G['ra']['Y']['r_ante'][:20, Tshock], label='RA')
    plt.plot(-G['ha']['Y']['r_ante'][:20, Tshock], label='HA')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.title('Impulse response on output to forward guidance')
    plt.xlabel(r"Year $(t)$")
    plt.ylabel('% deviation from ss')
    plt.legend(framealpha=0)
    plt.tight_layout()
    #plt.savefig('Export/FG_RA_HA_incomeinc_lec2.pdf', format='pdf', transparent=True)
    plt.show()


def figure_4(G, Tshock=10):
    fig = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot(-G['ra']['Y']['r_ante'][:20, Tshock], label='RA')
    plt.plot(-G['ha']['Y']['r_ante'][:20, Tshock], label='HA acyclical')
    plt.plot(-G['ha_counter']['Y']['r_ante'][:20, Tshock], label='HA counter-cyclical')
    plt.plot(-G['ha_pro']['Y']['r_ante'][:20, Tshock], label='HA pro-cyclical')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.title('Impulse response on output to forward guidance')
    plt.xlabel(r"Year $(t)$")
    plt.ylabel('% deviation from ss')
    plt.legend(framealpha=0)
    plt.tight_layout()
    #plt.savefig('Export/FG_RA_HA_incomeinc_lec2.pdf', format='pdf', transparent=True)
    plt.show()


def figure_5(irf, ss):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title('Output response')
    ax.plot(100 * irf['ha']['C'][0:20] / ss['ha']['Y'], label='HA (short)')
    ax.plot(100 * irf['ra']['C'][:20] / ss['ra']['Y'], linestyle = '--', label='RA')
    ax.plot(100 * irf['long']['C'][0:20] / ss['long']['Y'], label='HA (long, $\delta=0.95$)')
    ax.axhline(y=0, color='#808080', linestyle=':')
    ax.legend(framealpha=0)
    ax.set_ylabel('% deviation from ss')
    ax.set_xlabel(r"Year $(t)$")
    plt.show()


def figure_6(theta_list, irf_Y, irf_r_post):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
    ax1.set_title('Output response')
    for i, m in enumerate(theta_list):  ax1.plot(100 * irf_Y[i][0:20], label=fr'$\theta =$ %0.2f' % m)
    ax1.axhline(y=0, color='#808080', linestyle=':')
    ax1.set_ylabel('% deviation from ss')
    ax1.set_xlabel(r"Year $(t)$")
    ax1.legend(framealpha=0)
    ax2.set_title('Ex-post return')
    for i, m in enumerate(theta_list):  ax2.plot(100 * irf_r_post[i][0:20])
    ax2.axhline(y=0, color='#808080', linestyle=':')
    ax2.set_xlabel(r"Year $(t)$")
    ax2.set_ylabel('pp deviation from ss')
    plt.tight_layout()
    #plt.savefig('Export/fig_nom_assets.pdf', format='pdf', transparent=True)
    plt.show()
