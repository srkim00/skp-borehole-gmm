# Figures 2-11 and 13.
import re, pickle, logging, warnings
logging.disable(logging.WARNING); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pygmm
from pygmm import BooreStewartSeyhanAtkinson2014 as B
from config import WORK, FIG, FAC, MCAP

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
                     'legend.fontsize': 6.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'axes.grid': True, 'grid.color': '#d9d9d9',
                     'grid.linestyle': '--', 'grid.linewidth': 0.4, 'legend.frameon': True, 'legend.framealpha': 1.0,
                     'legend.edgecolor': '#444444', 'pdf.fonttype': 42, 'savefig.dpi': 300})

acc = pd.read_pickle(f'{WORK}/acc.pkl')
res_acc = pd.read_pickle(f'{WORK}/res_acc.pkl')
sm = pd.read_pickle(f'{WORK}/coeff_main_smoothed.pkl'); cf = pd.read_pickle(f'{WORK}/conv_factors.pkl'); cm4 = pd.read_pickle(f'{WORK}/coeff_m4.pkl')
stm = pd.read_pickle(f'{WORK}/station_mean_lnratio.pkl')
with open(f'{WORK}/final_effects.pkl', 'rb') as fh:
    final = pickle.load(fh)
ev = acc.groupby('event_id').agg(Mw=('Mw', 'first'), lat=('event_latitude_deg', 'first'), lon=('event_longitude_deg', 'first'))
pohang = ev[(ev.lat.between(35.9, 36.3)) & (ev.lon.between(129.2, 129.5))].index.tolist()
is_poh = acc.event_id.isin(pohang).values
tmax = 1.0 / (FAC * acc.highpass_hz.values); tmin = FAC / acc.lowpass_hz.values

PANELS = ['PGV_cm_s', 'PGA_g', 'SA(0.050)_g', 'SA(0.100)_g', 'SA(0.300)_g', 'SA(1.000)_g', 'SA(2.000)_g', 'SA(3.000)_g']
LABEL = {'PGV_cm_s': 'PGV (cm/s)', 'PGA_g': 'PGA (g)'}
for c in PANELS[2:]:
    LABEL[c] = c.replace('_g', ' (g)')
PERIOD = {im: (np.nan if im in ('PGV_cm_s', 'PGA_g') else float(re.search(r'SA\(([\d.]+)\)', im).group(1))) for im in sm.IM}
LETTERS = 'abcdefgh'


def usable(im):
    t = PERIOD[im]
    return np.ones(len(acc), bool) if np.isnan(t) else ((t <= tmax) & (t >= tmin))


def binned(x, y, edges, minn=5):
    cx, my, sy = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < hi)
        if k.sum() >= minn:
            cx.append(0.5 * (lo + hi)); my.append(y[k].mean()); sy.append(y[k].std())
    return np.array(cx), np.array(my), np.array(sy)


def model_ln(im, M, R):
    row = sm[sm.IM == im].iloc[0]
    t = PERIOD[im]
    outv = np.zeros(len(R))
    for i, Ri in enumerate(R):
        mod = B(pygmm.Scenario(mag=M, dist_jb=max(Ri, 0.0), v_s30=760, mechanism='SS', region='global'))
        if im == 'PGA_g':
            ref = mod.pga
        elif im == 'PGV_cm_s':
            ref = mod.pgv
        else:
            per = np.array(mod.periods); ref = np.array(mod.spec_accels)[np.argmin(abs(per - t))]
        outv[i] = np.log(ref) + row.a_s + row.a1_s * Ri + row.b1_s * min(MCAP, M)
    return outv


MGROUPS = [((3.0, 4.0), dict(marker='o', s=3, color='#8f8fd6', lw=0, label='$M_\\mathrm{W}$ 3–4')),
           ((4.0, 5.0), dict(marker='^', s=6, color='#2e7d32', lw=0, label='$M_\\mathrm{W}$ 4–5')),
           ((5.0, 6.0), dict(marker='s', s=7, facecolors='none', edgecolors='#c8963e', lw=0.6, label='$M_\\mathrm{W}$ 5–6'))]
RGROUPS = [((0.0, 150.0), dict(marker='s', s=5, facecolors='none', edgecolors='#c8963e', lw=0.5, label='$R_\\mathrm{JB}$ < 150 km')),
           ((150.0, 301.0), dict(marker='o', s=3, color='#8f8fd6', lw=0, label='$R_\\mathrm{JB}$ 150–300 km'))]
DEDGES = np.arange(0, 301, 50); MEDGES = np.arange(3.0, 6.01, 0.5)


def residual_figure(fname, get_y, xkey, ylabel, bincolor, groups, ylim=(-3.5, 3.5)):
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.5), sharey=True)
    handles = None
    for k, (ax, im) in enumerate(zip(axs.ravel(), PANELS)):
        y_all, m, mbin = get_y(im)
        x = (acc.Rjb_km.values if xkey == 'R' else acc.Mw.values)
        mm = m & ~is_poh
        for (lo, hi), st in groups:
            gkey = (acc.Mw.values if xkey == 'R' else acc.Rjb_km.values)
            kk = mm & (gkey >= lo) & (gkey < hi)
            ax.scatter(x[kk], y_all[kk], zorder=2, rasterized=True, **st)
        edges = DEDGES if xkey == 'R' else MEDGES
        mm = mbin & ~is_poh
        cx, my, sy = binned(x[mm], y_all[mm], edges)
        ax.errorbar(cx, my, yerr=sy, fmt='o', color=bincolor, ms=3.5, lw=0.8, capsize=1.5, zorder=4, label='Mean ± 1 SD in bins')
        ax.axhline(0, color='k', lw=0.7, zorder=6)
        ax.set_ylim(*ylim)
        if xkey == 'R':
            ax.set_xlim(0, 300); ax.set_xticks([0, 100, 200, 300])
        else:
            ax.set_xlim(2.9, 6.1); ax.set_xticks([3, 4, 5, 6])
        ax.text(0.97, 0.05, LABEL[im], transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        ax.text(-0.02, 1.02, f'({LETTERS[k]})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    for ax in axs[1]:
        ax.set_xlabel('Joyner–Boore distance (km)' if xkey == 'R' else 'Moment magnitude')
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    seen = {}; hh = []; ll = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = 1; hh.append(h); ll.append(l)
    fig.legend(hh, ll, loc='lower center', ncol=len(ll), bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=0.9, markerscale=1.6,
               fontsize=6.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f'{FIG}/{fname}.pdf'); plt.close(fig)


def residual_figure_model(fname, get_y, xkey, ylabel, groups, ylim=(-3.5, 3.5)):
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.5), sharey=True)
    for k, (ax, im) in enumerate(zip(axs.ravel(), PANELS)):
        y_all, m, mbin = get_y(im)
        x = (acc.Rjb_km.values if xkey == 'R' else acc.Mw.values)
        mm = m & ~is_poh
        for (lo, hi), st in groups:
            gkey = (acc.Mw.values if xkey == 'R' else acc.Rjb_km.values)
            kk = mm & (gkey >= lo) & (gkey < hi)
            ax.scatter(x[kk], y_all[kk], zorder=2, rasterized=True, **st)
        edges = DEDGES if xkey == 'R' else MEDGES
        mm = mbin & ~is_poh
        cx, my, sy = binned(x[mm], y_all[mm], edges)
        ax.errorbar(cx, my, yerr=sy, fmt='o', color='k', mfc='white', mew=0.8, ms=3.6, lw=0.8, capsize=1.5, zorder=5,
                    label='Mean ± 1 SD in bins, Pohang sequence excluded')
        kp = m & is_poh
        cxp, myp, syp = binned(x[kp], y_all[kp], edges)
        off = 8.0 if xkey == 'R' else 0.08
        ax.errorbar(cxp + off, myp, yerr=syp, fmt='D', color='#c0272d', mfc='#c0272d', mec='#6e0e14', mew=0.4, ms=2.8, lw=0.6, capsize=1.0,
                    zorder=4, label='Pohang sequence (not used in regression), mean ± 1 SD')
        ax.axhline(0, color='k', lw=0.7, zorder=6)
        ax.set_ylim(*ylim)
        if xkey == 'R':
            ax.set_xlim(0, 300); ax.set_xticks([0, 100, 200, 300])
        else:
            ax.set_xlim(2.9, 6.1); ax.set_xticks([3, 4, 5, 6])
        ax.text(0.97, 0.05, LABEL[im], transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        ax.text(-0.02, 1.02, f"({LETTERS[k]})", transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel('Joyner–Boore distance (km)' if xkey == 'R' else 'Moment magnitude')
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    h, l = axs[0, 0].get_legend_handles_labels()
    glab = [st['label'] for _, st in groups]
    gh = [h[l.index(s)] for s in glab]
    bl = [s for s in l if s not in glab]; bh = [h[l.index(s)] for s in bl]
    kw = dict(loc='lower center', handletextpad=0.3, columnspacing=1.2, markerscale=1.6, fontsize=6.5)
    fig.legend(gh, glab, ncol=len(gh), bbox_to_anchor=(0.5, 0.043), **kw)
    fig.legend(bh, bl, ncol=len(bh), bbox_to_anchor=(0.5, -0.012), **kw)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(f'{FIG}/{fname}.pdf'); plt.close(fig)


def y_uncorr(im):
    return res_acc[im].values, usable(im), usable(im)


residual_figure('figure_2', y_uncorr, 'R', 'ln(residual)', '#1f3fa8', MGROUPS)
residual_figure('figure_3', y_uncorr, 'M', 'ln(residual)', '#1f3fa8', RGROUPS)


def y_corr(im):
    return final[im]['corrected'].values, usable(im), final[im]['mask']


residual_figure_model('figure_4', y_corr, 'R', 'ln(residual)', MGROUPS)
residual_figure_model('figure_5', y_corr, 'M', 'ln(residual)', RGROUPS)


def fig_within():
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.4), sharey=True)
    for k, (ax, im) in enumerate(zip(axs.ravel(), PANELS)):
        fe = final[im]; m = fe['mask']
        w = fe['eps'].values + fe['dS'].reindex(acc.station_code.values[m]).values
        x = acc.Rjb_km.values[m]
        ax.scatter(x, w, s=2, color='#9a9a9a', lw=0, rasterized=True, zorder=2)
        cx, my, sy = binned(x, w, DEDGES)
        ax.errorbar(cx, my, yerr=sy, fmt='o', color='#c0272d', ms=3.5, lw=0.8, capsize=1.5, zorder=4)
        ax.axhline(0, color='k', lw=0.7, zorder=6); ax.set_ylim(-3, 3); ax.set_xlim(0, 300); ax.set_xticks([0, 100, 200, 300])
        ax.text(0.97, 0.05, LABEL[im], transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        ax.text(-0.02, 1.02, f'({LETTERS[k]})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel('Joyner–Boore distance (km)')
    for ax in axs[:, 0]:
        ax.set_ylabel('Within-event residual (ln)')
    fig.tight_layout(); fig.savefig(f'{FIG}/figure_6.pdf'); plt.close(fig)


def fig_between():
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.4), sharey=True)
    for k, (ax, im) in enumerate(zip(axs.ravel(), PANELS)):
        fe = final[im]; eta = fe['eta']; Mw = ev.Mw.reindex(eta.index).values
        ax.scatter(Mw, eta.values, s=5, color='#9a9a9a', lw=0, zorder=2)
        cx, my, sy = binned(Mw, eta.values, MEDGES, minn=2)
        ax.errorbar(cx, my, yerr=sy, fmt='o', color='#c0272d', ms=3.5, lw=0.8, capsize=1.5, zorder=4)
        ax.axhline(0, color='k', lw=0.7, zorder=6); ax.set_ylim(-2, 2); ax.set_xlim(2.9, 6.1); ax.set_xticks([3, 4, 5, 6])
        ax.text(0.97, 0.05, LABEL[im], transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        ax.text(-0.02, 1.02, f'({LETTERS[k]})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel('Moment magnitude')
    for ax in axs[:, 0]:
        ax.set_ylabel('Between-event residual (ln)')
    fig.tight_layout(); fig.savefig(f'{FIG}/figure_7.pdf'); plt.close(fig)


fig_within(); fig_between()


def fig_hist():
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.3), sharey=True)
    for k, (ax, im) in enumerate(zip(axs.ravel(), PANELS)):
        r = np.exp(stm[im].dropna().values); c = cf[cf.IM == im].iloc[0]
        ax.hist(r, bins=np.arange(0.0, 1.65, 0.1), color='#8c8c8c', edgecolor='k', lw=0.4)
        ax.axvline(c.CF, color='#c0272d', ls='--', lw=1.0)
        ax.set_xlim(0, 1.6); ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.text(0.96, 0.95, f'{LABEL[im].split(" (")[0]}\nfactor = {c.CF:.3f}\n{int(c.n_stations)} stations', transform=ax.transAxes, ha='right', va='top', fontsize=6.5)
        ax.text(-0.02, 1.02, f'({LETTERS[k]})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel('Number of stations')
    fig.supxlabel('Station-mean broadband-to-accelerometer ground-motion ratio', fontsize=7)
    fig.tight_layout(); fig.savefig(f'{FIG}/figure_10.pdf'); plt.close(fig)


fig_hist()


def fig_cf():
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    c = cf[cf['T'].notna()].sort_values('T')
    lo = np.exp(np.log(c.CF) - c.sd_ln_station); hi = np.exp(np.log(c.CF) + c.sd_ln_station)
    ax.fill_between(c['T'], lo, hi, color='#bfbfbf', alpha=0.6, lw=0, label='±1 SD of station means')
    ax.plot(c['T'], c.CF, color='k', lw=1.4, label='All stations')
    ax.plot(c['T'], c.CF_bb100, color='#1f3fa8', lw=1.0, label='Broadband sensor at 99–106 m')
    ax.plot(c['T'], c.CF_bb65, color='#c0272d', lw=1.0, label='Broadband sensor at 60–67 m')
    for im, xpos in [('PGV_cm_s', 0.0155), ('PGA_g', 0.0265)]:
        r = cf[cf.IM == im].iloc[0]
        ax.errorbar(xpos, r.CF, yerr=[[r.CF - np.exp(np.log(r.CF) - r.sd_ln_station)], [np.exp(np.log(r.CF) + r.sd_ln_station) - r.CF]], fmt='o', color='k', ms=3, lw=0.8, capsize=1.5)
    ax.axhline(1.0, color='#777777', lw=0.6)
    ax.set_xscale('log'); ax.set_xlim(0.0135, 3.3); ax.set_ylim(0.3, 1.2)
    ticks = [0.0155, 0.0265, 0.05, 0.1, 0.2, 0.5, 1, 2, 3]
    ax.set_xticks(ticks); ax.set_xticklabels(['PGV', 'PGA', '0.05', '0.1', '0.2', '0.5', '1', '2', '3'])
    ax.set_xticks([], minor=True); ax.axvline(0.0315, color='#777777', lw=0.5, ls=':')
    ax.set_xlabel('Period (s)'); ax.set_ylabel('Conversion factor (broadband / accelerometer)')
    ax.legend(loc='lower right', fontsize=5.5, ncol=1, handlelength=1.6, handletextpad=0.5, borderpad=0.4, labelspacing=0.3)
    fig.tight_layout(); fig.savefig(f'{FIG}/figure_11.pdf'); plt.close(fig)


fig_cf()


def fig_atten():
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 4.6))
    Rc = np.logspace(0, np.log10(300), 60)
    cmap = plt.get_cmap('viridis', 6); norm = matplotlib.colors.BoundaryNorm(np.arange(3.0, 6.01, 0.5), 6)
    for k, (ax, im) in enumerate(zip(axs.ravel(), PANELS)):
        m = usable(im) & ~is_poh
        sc = ax.scatter(acc.Rjb_km.values[m], acc[im].values[m], c=acc.Mw.values[m], cmap=cmap, norm=norm, s=3, lw=0, alpha=0.7, rasterized=True, zorder=2)
        for M, ls in [(3.25, '-'), (4.25, '-'), (5.25, '-'), (6.25, '--')]:
            ax.plot(Rc, np.exp(model_ln(im, M, Rc)), color='k', lw=0.9, ls=ls, zorder=3)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1, 300)
        ax.text(0.03, 0.05, LABEL[im], transform=ax.transAxes, ha='left', va='bottom', fontsize=7)
        ax.text(-0.02, 1.02, f'({LETTERS[k]})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
        if k == 0:
            for M in [6.25, 5.25, 4.25, 3.25]:
                yv = np.exp(model_ln(im, M, np.array([1.3])))[0]
                ax.text(1.3, yv / 1.8, f'$M_\\mathrm{{W}}$ {M}', fontsize=5.5, va='top', ha='left',
                        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.8))
    for ax in axs[1]:
        ax.set_xlabel('Joyner–Boore distance (km)')
    for ax in axs[:, 0]:
        ax.set_ylabel('Peak ground motion')
    fig.tight_layout(rect=(0, 0.05, 0.915, 1))
    cax = fig.add_axes([0.93, 0.2, 0.012, 0.6]); cb = fig.colorbar(sc, cax=cax); cb.set_label('Moment magnitude', fontsize=7)
    custom = [Line2D([0], [0], color='k', lw=0.9, ls='-', label='Model, $M_\\mathrm{W}$ 3.25, 4.25, and 5.25 (within the magnitude range of the data)'), Line2D([0], [0], color='k', lw=0.9, ls='--', label='Model, $M_\\mathrm{W}$ 6.25 (extrapolation)')]
    fig.legend(handles=custom, loc='lower center', ncol=2, bbox_to_anchor=(0.46, -0.005), fontsize=6)
    fig.savefig(f'{FIG}/figure_13.pdf'); plt.close(fig)


fig_atten()


def fig_coeff():
    s = sm[sm['T'].notna()].sort_values('T')
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.2))
    for ax, c, lab in zip(axs.ravel()[:3], ['a', 'a1', 'b1'], ['$a$', '$a_1$ (per km)', '$b_1$']):
        ax.fill_between(s['T'], s[c] - s['se_' + c], s[c] + s['se_' + c], color='#bfbfbf', alpha=0.7, lw=0, label='±1 standard error')
        ax.plot(s['T'], s[c], color='#8f8fd6', lw=0.7, label='Estimated at each period', zorder=4)
        ax.plot(s['T'], s[c + '_s'], color='k', lw=1.6, label='Smoothed (adopted)', zorder=3)
        ca = cm4[cm4['T'].notna()].sort_values('T')
        ax.plot(ca['T'], ca[c], color='#c0272d', lw=0.8, ls='--', label='$M_\\mathrm{W}$ ≥ 4 events only (not adopted)', zorder=4)
        ax.set_xscale('log'); ax.set_xlim(0.03, 3.3); ax.set_ylabel(lab); ax.axhline(0, color='#777777', lw=0.5)
    ax = axs.ravel()[3]
    for c, lab, col in [('tau_f', 'τ (between-event)', '#c0272d'), ('phiS2S_f', '$φ_\\mathrm{S2S}$ (station-to-station)', '#1f3fa8'), ('phiSS_f', '$φ_\\mathrm{SS}$ (single-station within-event)', '#2e7d32'), ('phi_f', 'φ (within-event)', '#8f8fd6'), ('sigma_f', 'σ (total)', 'k')]:
        ax.plot(s['T'], s[c], color=col, lw=1.1, label=lab)
    ax.set_xscale('log'); ax.set_xlim(0.03, 3.3); ax.set_ylim(0, 0.9); ax.set_ylabel('Standard deviation (ln)')
    hd, ld = ax.get_legend_handles_labels()
    ax = axs.ravel()[4]
    ax.plot(s['T'], s['n'], color='k', lw=1.2, label='Records')
    ax.plot(s['T'], s['n_events'], color='#c0272d', lw=1.2, label='Events')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.03, 3.3); ax.set_ylim(1, 20000); ax.set_ylabel('Number used in regression')
    ax.legend(loc='lower left', fontsize=5.2, labelspacing=0.25, handlelength=1.6, borderpad=0.3)
    axs.ravel()[5].axis('off')
    for k, ax in enumerate(axs.ravel()[:5]):
        ax.set_xlabel('Period (s)'); ax.text(-0.02, 1.02, f'({LETTERS[k]})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    h, l = axs.ravel()[0].get_legend_handles_labels()
    leg1 = axs.ravel()[5].legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=6.2, title='Panels (a)–(c)', title_fontsize=6.2)
    axs.ravel()[5].add_artist(leg1)
    axs.ravel()[5].legend(hd, ld, loc='lower center', bbox_to_anchor=(0.5, -0.08), fontsize=6.2, title='Panel (d)', title_fontsize=6.2)
    fig.tight_layout(); fig.savefig(f'{FIG}/figure_8.pdf'); plt.close(fig)


fig_coeff()


def fig_spectra():
    s = sm[sm['T'].notna()].sort_values('T'); Ts = s['T'].values
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    cols = {4.0: '#2e7d32', 5.0: '#1f3fa8', 6.0: '#c0272d'}
    for ax, R in zip(axs, [20.0, 100.0]):
        for M in [4.0, 5.0, 6.0]:
            mod = B(pygmm.Scenario(mag=M, dist_jb=R, v_s30=760, mechanism='SS', region='global'))
            per = np.array(mod.periods); ref = np.array([np.array(mod.spec_accels)[np.argmin(abs(per - t))] for t in Ts])
            accl = ref * np.exp(s.a_s.values + s.a1_s.values * R + s.b1_s.values * min(MCAP, M))
            cfv = cf.set_index('IM').reindex(s.IM).CF.values
            ax.plot(Ts, accl, color=cols[M], lw=1.3, label=f'ACC level, $M_\\mathrm{{W}}$ {M:.0f}' + (' (extrapolation)' if M > 5.8 else ''))
            ax.plot(Ts, accl * cfv, color=cols[M], lw=1.0, ls='--', label=f'BB level, $M_\\mathrm{{W}}$ {M:.0f}')
            ax.plot(Ts, ref, color=cols[M], lw=0.7, ls=':', label=f'BSSA14 ($V_\\mathrm{{S30}}$ = 760 m/s), $M_\\mathrm{{W}}$ {M:.0f}')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.03, 3.3)
        ax.set_xlabel('Period (s)'); ax.text(0.03, 0.05, f'$R_\\mathrm{{JB}}$ = {R:.0f} km', transform=ax.transAxes, fontsize=7)
    axs[0].set_ylabel('5%-damped spectral acceleration (g)')
    for k, ax in enumerate(axs):
        ax.text(-0.02, 1.02, f'({LETTERS[k]})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.005), fontsize=6)
    fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig(f'{FIG}/figure_9.pdf'); plt.close(fig)


fig_spectra()
