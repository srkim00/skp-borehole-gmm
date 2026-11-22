# Figures 12 and S4.
import pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from config import OUT, WORK, FIG, TOL

STN = ['GKP2', 'CHRB', 'CIGB', 'EURB']
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'axes.linewidth': 0.6, 'pdf.fonttype': 42,
                     'savefig.dpi': 300, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'xtick.minor.size': 0, 'ytick.minor.size': 0, 'axes.grid': True,
                     'grid.linewidth': 0.4})
p1 = pd.read_pickle(f'{WORK}/pairs_kept.pkl')
stm = pd.read_pickle(f'{WORK}/station_mean_lnratio.pkl')
with open(f'{WORK}/screen_detail.pkl', 'rb') as fh:
    detail = pickle.load(fh)
tfz = np.load(f'{WORK}/onedim_tf.npz'); F = tfz['freq']; Tf = 1 / F
SA = [c for c in stm.columns if c.startswith('SA(')]; T = np.array([float(c[3:-3]) for c in SA])
q = 'Q100'
syn = pd.read_csv(f'{OUT}/onedim_syn_records_{q}.csv'); synm = pd.read_csv(f'{OUT}/onedim_syn_station_means_{q}.csv')
WIN = (T >= 0.08 - 1e-9) & (T <= 0.4 + 1e-9)
TPLOT = T <= 1.0 + 1e-9                      # periods drawn (0.035-1.0 s)
TFWIN = (Tf >= 0.03) & (Tf <= 1.0)           # transfer functions drawn between 0.03 and 1.0 s
RED, BLUE, PINK, LBLUE, BAND = '#c0272d', '#1d4593', '#e39699', '#9fb8e0', '#7fa3dc'


def observed(stn):
    d = p1[p1.station_code == stn]
    M = np.full((len(d), len(SA)), np.nan)
    for j, (im, t) in enumerate(zip(SA, T)):
        ok = (t <= d.tmax_pair.values + TOL) & (t >= d.tmin_pair.values - TOL)
        rem = set(detail[im]['range']) | set(detail[im]['station']) | set(detail[im]['record'])
        ok &= ~d.index.isin(list(rem))
        M[ok, j] = (d[im + '_b'] / d[im + '_a']).values[ok]
    keep = np.isfinite(M).any(axis=1)
    pgv = (d['PGV_cm_s_b'] / d['PGV_cm_s_a']).to_numpy(copy=True); pga = (d['PGA_g_b'] / d['PGA_g_a']).to_numpy(copy=True)
    for im, arr in (('PGV_cm_s', pgv), ('PGA_g', pga)):
        rem = set(detail[im]['range']) | set(detail[im]['station']) | set(detail[im]['record'])
        arr[d.index.isin(list(rem))] = np.nan
    return M[keep], pgv[keep], pga[keep]


def synthetic(stn):
    d = syn[syn.station == stn]
    M = np.full((len(d), len(T)), np.nan)
    for j, t in enumerate(T):
        ok = (t <= d.T_max.values + TOL) & (t >= d.T_min.values - TOL)
        M[ok, j] = (d['syn_%g' % t] / d['acc_%g' % t]).values[ok]
    return M


def minima(M):
    pts = []
    for row in M:
        w = row[WIN]
        if np.isfinite(w).mean() >= 0.9:
            k = np.nanargmin(w); pts.append((T[WIN][k], w[k]))
    return np.array(pts)


def axes_at(fig, W, H, x0, top, w, h):
    return fig.add_axes([x0 / W, 1 - (top + h) / H, w / W, h / H])


def figure12():
    W, H = 518.4, 446.4
    fig = plt.figure(figsize=(W / 72, H / 72))
    pos = [(38.88, 14.73), (282.96, 14.73), (38.88, 207.42), (282.96, 207.42)]
    XPGV, XPGA, XSEP = 0.016, 0.0225, 0.0295
    for k, (stn, (x0, top)) in enumerate(zip(STN, pos)):
        ax = axes_at(fig, W, H, x0, top, 230.26, 167.56)
        M, pgv, pga = observed(stn)
        for row in M:
            ax.plot(T[TPLOT], row[TPLOT], color=PINK, lw=0.4, zorder=2)
        om = np.exp(stm.loc[stn, SA].astype(float).values)
        mp = minima(M)
        ax.plot(mp[:, 0], mp[:, 1], 'o', color=RED, ms=2.8, alpha=0.55, mec='none', zorder=5)
        ax.plot(T[TPLOT], om[TPLOT], color=RED, lw=1.6, zorder=6)
        rms = tfz[f'{stn}_{q}_rms']
        ax.fill_between(Tf[TFWIN], np.percentile(rms, 16, axis=0)[TFWIN], np.percentile(rms, 84, axis=0)[TFWIN],
                        color=BAND, alpha=0.4, lw=0, zorder=7)
        ax.plot(Tf[TFWIN], np.median(rms, axis=0)[TFWIN], color=BLUE, lw=1.5, alpha=0.75, zorder=8)
        for x, v in ((XPGV, pgv), (XPGA, pga)):
            v = v[np.isfinite(v)]
            ax.plot(np.full(len(v), x), v, 'o', color=PINK, ms=1.8, mec='none', zorder=3)
        for x, im in ((XPGV, 'PGV_cm_s'), (XPGA, 'PGA_g')):
            ax.plot([x], [np.exp(stm.loc[stn, im])], 'o', color=RED, ms=3.6, mew=1.0, zorder=6)
        ax.axhline(1, color='#777777', lw=0.6, zorder=1)
        ax.axvline(XSEP, color='#777777', lw=0.5, ls=':', zorder=1)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.014, 1.1); ax.set_ylim(0.2, 5)
        ax.grid(True, color='#d9d9d9', ls='--', lw=0.4)
        ax.set_xticks([XPGV, XPGA, 0.05, 0.1, 0.2, 0.5, 1])
        ax.set_yticks([0.2, 0.3, 0.5, 1, 2, 3, 5])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if k >= 2:
            ax.set_xticklabels(['PGV', 'PGA', '0.05', '0.1', '0.2', '0.5', '1'], fontsize=6.2)
            ax.set_xlabel('Period (s)', fontsize=7.5)
        else:
            ax.set_xticklabels([])
        if k % 2 == 0:
            ax.set_yticklabels(['0.2', '0.3', '0.5', '1', '2', '3', '5'], fontsize=6.5)
            ax.set_ylabel('Broadband-to-accelerometer ratio', fontsize=7.5)
        else:
            ax.set_yticklabels([])
        ax.text(0.97, 0.96, stn, transform=ax.transAxes, ha='right', va='top', fontsize=8)
        ax.text(0.0, 1.012, '(%s)' % 'abcd'[k], transform=ax.transAxes, ha='left', va='bottom', fontsize=8)
    hs = [Line2D([], [], color=PINK, lw=0.7), Line2D([], [], ls='none', marker='o', color=RED, ms=3.4, alpha=0.55, mec='none'),
          Line2D([], [], color=RED, lw=1.6), Line2D([], [], color=BLUE, lw=1.5, alpha=0.75), Patch(facecolor=BAND, alpha=0.4, lw=0)]
    ls = ['Observed SA ratio, individual record pairs', 'Minimum of each observed record, 0.08–0.4 s', 'Observed SA ratio, station mean',
          '1D transfer function, ensemble median', '1D transfer function, 16th–84th percentile range of the ensemble']
    leg = fig.legend(hs, ls, loc='lower center', bbox_to_anchor=(0.53, 0.0), ncol=2, fontsize=6.8, handlelength=3.0,
                     frameon=True, framealpha=0.8, edgecolor='#444444')
    leg.get_frame().set_linewidth(0.6)
    return fig


def figureS4():
    W, H = 518.4, 360.0
    fig = plt.figure(figsize=(W / 72, H / 72))
    xs = [44.06, 163.56, 283.07, 402.57]
    for k, stn in enumerate(STN):
        M, _, _ = observed(stn); S = synthetic(stn)
        om = np.exp(stm.loc[stn, SA].astype(float).values); sm_ = synm[synm.station == stn]
        for r, (X, thin, thick, mean_T, mean_v, top) in enumerate(((M, '#e3a0a3', RED, T[TPLOT], om[TPLOT], 30.6),
                                                                    (S, LBLUE, BLUE, sm_['T'].values, sm_.ratio.values, 176.86))):
            ax = axes_at(fig, W, H, xs[k], top, 110.65, 107.54)
            for row in X:
                ax.plot(T[TPLOT], row[TPLOT], color=thin, lw=0.45, zorder=2)
            ax.plot(mean_T, mean_v, color=thick, lw=1.6, zorder=6)
            mp = minima(X)
            ax.plot(mp[:, 0], mp[:, 1], 'o', mfc='white', mec=thick, ms=3.0, mew=0.7, zorder=5)
            ax.axhline(1, color='#777777', lw=0.6, zorder=1)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.033, 1.05); ax.set_ylim(0.1, 10)
            ax.grid(True, color='#d0d0d0', ls=(0, (3, 2)), lw=0.4)
            ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1]); ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if r == 1:
                ax.set_xticklabels(['0.05', '0.1', '0.2', '0.5', '1'], fontsize=7.5); ax.set_xlabel('Period (s)', fontsize=7.5)
            else:
                ax.set_xticklabels([])
            if k == 0:
                ax.set_yticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10'], fontsize=7.5)
            else:
                ax.set_yticklabels([])
            ax.text(0.97, 0.96, stn, transform=ax.transAxes, ha='right', va='top', fontsize=8)
            ax.text(0.0, 1.02, '(%s)' % ('abcd' if r == 0 else 'efgh')[k], transform=ax.transAxes, ha='left', va='bottom', fontsize=8)
            if k == 0:
                ax.text(0.0, 1.115, 'Observed' if r == 0 else
                        r'Synthetic, oblique incidence of SH and SV waves, $Q_\mathrm{S}$ = $V_\mathrm{S}$/%s' % q[1:],
                        transform=ax.transAxes, ha='left', va='bottom', fontsize=8, fontweight='bold')
    fig.text(5.2 / W, 1 - 157.5 / H, 'Broadband-to-accelerometer ratio', rotation=90, ha='left', va='center', fontsize=7.5)
    hs = [(Line2D([], [], color='#e3a0a3', lw=0.8), Line2D([], [], color=LBLUE, lw=0.8)),
          (Line2D([], [], color=RED, lw=1.6), Line2D([], [], color=BLUE, lw=1.6)),
          (Line2D([], [], ls='none', marker='o', mfc='white', mec=RED, ms=3.5, mew=1.0),
           Line2D([], [], ls='none', marker='o', mfc='white', mec=BLUE, ms=3.5, mew=1.0))]
    leg = fig.legend(hs, ['SA ratio of individual records', 'Station-specific mean', 'Minimum of each record, 0.08–0.4 s'],
                     handler_map={tuple: HandlerTuple(ndivide=None)}, loc='lower center', bbox_to_anchor=(0.53, 0.0),
                     fontsize=7.0, handlelength=4.2, frameon=True, framealpha=0.8, edgecolor='black')
    leg.get_frame().set_linewidth(0.6)
    return fig


if __name__ == '__main__':
    fig = figure12(); fig.savefig(f'{FIG}/figure_12.pdf'); plt.close(fig)
    fig = figureS4(); fig.savefig(f'{FIG}/figure_S4.pdf'); plt.close(fig)
