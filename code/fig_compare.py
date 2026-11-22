# Figures 14-18.
import re, pickle, logging, warnings
logging.disable(logging.WARNING); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygmm
from pygmm import BooreStewartSeyhanAtkinson2014 as B
from config import WORK, FIG, FAC, MCAP
from previous_gmms import EMOLO15, KHNP15, JH22

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
                     'legend.fontsize': 6.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'axes.grid': True, 'grid.color': '#d9d9d9',
                     'grid.linestyle': '--', 'grid.linewidth': 0.4, 'legend.frameon': True, 'legend.framealpha': 1.0,
                     'legend.edgecolor': '#444444', 'pdf.fonttype': 42, 'savefig.dpi': 300})

acc = pd.read_pickle(f'{WORK}/acc.pkl')
sm = pd.read_pickle(f'{WORK}/coeff_main_smoothed.pkl').set_index('IM')
with open(f'{WORK}/final_effects.pkl', 'rb') as fh:
    final = pickle.load(fh)
with open(f'{WORK}/stoch_psa.pkl', 'rb') as fh:
    stoch = pickle.load(fh)
ev = acc.groupby('event_id').agg(Mw=('Mw', 'first'), lat=('event_latitude_deg', 'first'), lon=('event_longitude_deg', 'first'), dep=('event_depth_km', 'first'))
pohang = ev[(ev.lat.between(35.9, 36.3)) & (ev.lon.between(129.2, 129.5))].index.tolist()
is_poh = acc.event_id.isin(pohang).values
tmax = 1.0 / (FAC * acc.highpass_hz.values); tmin = FAC / acc.lowpass_hz.values
PERIOD = {im: (np.nan if im in ('PGV_cm_s', 'PGA_g') else float(re.search(r'SA\(([\d.]+)\)', im).group(1))) for im in sm.index}
LETTERS = 'abcdefgh'
LABEL = {'PGV_cm_s': 'PGV (cm/s)', 'PGA_g': 'PGA (g)'}
for im in sm.index:
    LABEL.setdefault(im, im.replace('_g', ' (g)'))


def usable(im):
    t = PERIOD[im]
    return np.ones(len(acc), bool) if np.isnan(t) else ((t <= tmax) & (t >= tmin))


def model_ln(im, M, R):
    M = np.atleast_1d(np.asarray(M, float)); R = np.atleast_1d(np.asarray(R, float))
    if M.size == 1 and R.size > 1:
        M = np.full(R.shape, M[0])
    if R.size == 1 and M.size > 1:
        R = np.full(M.shape, R[0])
    r = sm.loc[im]
    ref = np.array([ref_bssa(im, m, x) for m, x in zip(M, R)])
    return np.log(ref) + r.a_s + r.a1_s * R + r.b1_s * np.minimum(MCAP, M)


_cache = {}


def ref_bssa(im, M, R):
    key = (im, round(float(M), 4), round(float(R), 3))
    if key in _cache:
        return _cache[key]
    mod = B(pygmm.Scenario(mag=float(M), dist_jb=float(R), v_s30=760, mechanism='SS', region='global'))
    if im == 'PGV_cm_s':
        v = mod.pgv
    elif im == 'PGA_g':
        v = mod.pga
    else:
        per = np.array(mod.periods); v = np.array(mod.spec_accels)[np.argmin(abs(per - PERIOD[im]))]
    _cache[key] = float(v)
    return float(v)


def ref_bssa_arr(im, M, R):
    return np.array([ref_bssa(im, m, x) for m, x in zip(np.broadcast_to(M, np.shape(R)) if np.ndim(M) == 0 else M, R)])


def imt_name(im):
    return {'PGV_cm_s': 'PGV', 'PGA_g': 'PGA'}.get(im, im.replace('_g', ''))


def binned(x, y, edges, minn=1):
    cx, my, sy = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < hi)
        if k.sum() >= minn:
            cx.append(x[k].mean()); my.append(y[k].mean()); sy.append(y[k].std() if k.sum() > 1 else 0.0)
    return np.array(cx), np.array(my), np.array(sy)


def prev_models(im, Mw, Rjb, Repi, Rhypo, curve=False, depth=10.0):
    name = imt_name(im)
    Mw = np.asarray(Mw, float); Rjb = np.asarray(Rjb, float)
    outd = {}
    outd['BSSA14'] = ref_bssa_arr(im, Mw, Rjb)
    ml = (1.086 * Mw - 0.4772 - 0.3262) / 0.9234     # ML on the KMA scale used before July 2019
    outd['EMOLO15'] = np.asarray(EMOLO15(name, np.asarray(Repi, float), ml)._compute())
    if curve:
        outd['KHNP15'] = np.asarray(KHNP15(name, np.asarray(Repi, float), Mw, Repi=True, depth=depth)._compute())
        outd['JH22'] = np.asarray(JH22(name, np.asarray(Repi, float), Mw, Repi=True, depth=depth)._compute())
    else:
        outd['KHNP15'] = np.asarray(KHNP15(name, np.asarray(Rhypo, float), Mw)._compute())
        outd['JH22'] = np.asarray(JH22(name, np.asarray(Rhypo, float), Mw)._compute())
    return outd


STY = {'This study': dict(color='k', marker='o', ls='-', lw=1.4), 'BSSA14': dict(color='#c0272d', marker='s', ls='-.', lw=0.9),
       'EMOLO15': dict(color='#1f3fa8', marker='^', ls=':', lw=0.9), 'KHNP15': dict(color='#7b2d8e', marker='*', ls='--', lw=0.9),
       'JH22': dict(color='#2e7d32', marker='v', ls='--', lw=0.9)}
ORDER = ['This study', 'BSSA14', 'EMOLO15', 'KHNP15', 'JH22']


def resid_panel(ax, R, obs, preds, edges, offset=0.0):
    for j, name in enumerate(ORDER):
        res = np.log(obs) - np.log(preds[name])
        cx, my, sy = binned(R, res, edges)
        st = STY[name]
        ax.errorbar(cx + (j - 2) * offset, my, yerr=sy, fmt=st['marker'], color=st['color'], ms=3.2 if name != 'KHNP15' else 4.5, lw=0.7, capsize=1.2,
                    ls=st['ls'], zorder=5 if name == 'This study' else 3, label=name)
    ax.axhline(0, color='k', lw=0.6); ax.set_ylim(-2.5, 2.5)


def curve_panel(ax, im, M, dist, depth, obsR=None, obsY=None, obs_color='#777777', dashed_beyond=300.0):
    preds = prev_models(im, np.full(dist.shape, M), dist, dist, None, curve=True, depth=depth)
    ours = np.exp(model_ln(im, M, dist))
    k = dist <= dashed_beyond
    ax.plot(dist[k], ours[k], color='k', lw=1.4, zorder=5, label='This study')
    if (~k).any():
        ax.plot(dist[~k], ours[~k], color='k', lw=1.4, ls='--', zorder=5)
    for name in ORDER[1:]:
        st = STY[name]
        ax.plot(dist, preds[name], color=st['color'], lw=st['lw'], ls=st['ls'], label=name, zorder=3)
    if obsR is not None:
        ax.scatter(obsR, obsY, s=9, facecolors='none', edgecolors=obs_color, lw=0.5, zorder=2, rasterized=True)
    ax.set_xscale('log'); ax.set_yscale('log')


def fig_pga():
    im = 'PGA_g'
    fig, axs = plt.subplots(2, 3, figsize=(6.2, 4.4), gridspec_kw={'height_ratios': [1, 1.8]})
    main = acc[~is_poh]
    for col, M in enumerate([3.25, 4.25, 5.25]):
        sub = main[(main.Mw >= M - 0.25) & (main.Mw <= M + 0.25)]
        preds = prev_models(im, sub.Mw.values, sub.Rjb_km.values, sub.Repi_km.values, sub.Rhypo_km.values)
        preds['This study'] = np.exp(model_ln(im, sub.Mw.values, sub.Rjb_km.values))
        resid_panel(axs[0, col], sub.Rjb_km.values, sub[im].values, preds, np.arange(0, 301, 50), offset=3.0)
        axs[0, col].set_xlim(0, 300); axs[0, col].set_xticks([0, 100, 200, 300])
        dist = np.logspace(0, np.log10(300), 80)
        curve_panel(axs[1, col], im, M, dist, 10.0, sub.Rjb_km.values, sub[im].values)
        axs[1, col].set_xlim(1, 300); axs[1, col].set_ylim(1e-6, 1e0)
        axs[0, col].set_title(f'$M_\\mathrm{{W}}$ {M} ± 0.25 ({len(sub):,} records)', fontsize=6.5, loc='right', pad=3)
    for k, ax in enumerate(axs.ravel()):
        ax.text(-0.02, 1.02, f"({LETTERS[k]})", transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel('Joyner–Boore distance (km)')
    axs[0, 0].set_ylabel('ln(residual)'); axs[1, 0].set_ylabel('PGA (g)')
    for ax in list(axs[0, 1:]) + list(axs[1, 1:]):
        ax.tick_params(labelleft=False)
    hh, ll = axs[1, 0].get_legend_handles_labels()
    fig.legend(hh, ll, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f'{FIG}/figure_14.pdf'); plt.close(fig)


def fig_event(eqid, fname, ylim):
    ims = ['SA(0.100)_g', 'SA(0.200)_g', 'SA(0.500)_g', 'SA(1.000)_g']
    sub0 = acc[acc.event_id == eqid]
    M = float(sub0.Mw.iloc[0]); depth = float(sub0.event_depth_km.iloc[0])
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 4.4), gridspec_kw={'height_ratios': [1, 1.8]})
    for col, im in enumerate(ims):
        u = usable(im)[acc.index.get_indexer(sub0.index)]
        sub = sub0[u]
        preds = prev_models(im, sub.Mw.values, sub.Rjb_km.values, sub.Repi_km.values, sub.Rhypo_km.values)
        preds['This study'] = np.exp(model_ln(im, sub.Mw.values, sub.Rjb_km.values))
        resid_panel(axs[0, col], sub.Rjb_km.values, sub[im].values, preds, np.arange(0, 301, 50), offset=3.0)
        axs[0, col].set_xlim(0, 300); axs[0, col].set_xticks([0, 100, 200, 300])
        dist = np.logspace(0, np.log10(300), 80)
        curve_panel(axs[1, col], im, M, dist, depth, sub.Rjb_km.values, sub[im].values)
        axs[1, col].set_xlim(1, 300); axs[1, col].set_ylim(*ylim)
        axs[0, col].set_title(f'{imt_name(im)}, {len(sub)} records', fontsize=6.5, loc='right', pad=3)
        axs[1, col].text(0.03, 0.05, LABEL[im], transform=axs[1, col].transAxes, ha='left', va='bottom', fontsize=7)
    for k, ax in enumerate(axs.ravel()):
        ax.text(-0.02, 1.02, f'({LETTERS[k]})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel('Joyner–Boore distance (km)')
    axs[0, 0].set_ylabel('ln(residual)'); axs[1, 0].set_ylabel('5%-damped SA (g)')
    for ax in axs[0, 1:]:
        ax.tick_params(labelleft=False)
    for ax in axs[1, 1:]:
        ax.tick_params(labelleft=False)
    h, l = axs[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f'{FIG}/{fname}.pdf'); plt.close(fig)


PANELS = ['PGV_cm_s', 'PGA_g', 'SA(0.050)_g', 'SA(0.100)_g', 'SA(0.300)_g', 'SA(1.000)_g', 'SA(2.000)_g', 'SA(3.000)_g']


def fig_stoch(xkey, fname):
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.5), sharey=True)
    edges = np.arange(0, 301, 50) if xkey == 'R' else np.arange(3.0, 6.01, 0.5)
    x_all = acc.Rjb_km.values if xkey == 'R' else acc.Mw.values
    for k, (ax, im) in enumerate(zip(axs.ravel(), PANELS)):
        m = usable(im) & ~is_poh
        series = [('This study', final[im]['corrected'].values, dict(fmt='o-', color='#c0272d', ms=3.2, lw=0.9, zorder=5)),
                  ('Stochastic, WNA attenuation', np.log(acc[im].values) - np.log(stoch['WNA'][im].values), dict(fmt='s--', color='#1f3fa8', ms=3.0, lw=0.9, zorder=3)),
                  ('Stochastic, SKP attenuation', np.log(acc[im].values) - np.log(stoch['SKP'][im].values), dict(fmt='^--', color='#c8963e', ms=3.2, lw=0.9, zorder=4))]
        for j, (name, y, st) in enumerate(series):
            cx, my, sy = binned(x_all[m], y[m], edges, minn=5)
            off = (j - 1) * (4.0 if xkey == 'R' else 0.04)
            ax.errorbar(cx + off, my, yerr=sy, capsize=1.2, label=name, **st)
        ax.axhline(0, color='k', lw=0.6, zorder=1); ax.set_ylim(-3.0, 3.0)
        if xkey == 'R':
            ax.set_xlim(0, 300); ax.set_xticks([0, 100, 200, 300])
        else:
            ax.set_xlim(2.9, 6.1); ax.set_xticks([3, 4, 5, 6])
        ax.text(0.97, 0.05, LABEL[im], transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        ax.text(-0.02, 1.02, f'({LETTERS[k]})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel('Joyner–Boore distance (km)' if xkey == 'R' else 'Moment magnitude')
    for ax in axs[:, 0]:
        ax.set_ylabel('ln(residual)')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f'{FIG}/{fname}.pdf'); plt.close(fig)


if __name__ == '__main__':
    fig_pga()
    fig_event('kr2019202020418', 'figure_15', (1e-7, 1e-1))
    fig_event('kr2016256113254', 'figure_16', (1e-5, 1e0))
    fig_stoch('R', 'figure_17')
    fig_stoch('M', 'figure_18')
