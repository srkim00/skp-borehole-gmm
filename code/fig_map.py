# Figure 1 and Figure S2.
import logging, warnings
logging.disable(logging.WARNING); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.basemap import Basemap
from config import WORK, FIG, FAC

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
                     'legend.fontsize': 6.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'axes.grid': True, 'grid.color': '#d9d9d9',
                     'grid.linestyle': '--', 'grid.linewidth': 0.4, 'legend.frameon': True, 'legend.framealpha': 1.0,
                     'legend.edgecolor': '#444444', 'pdf.fonttype': 42, 'savefig.dpi': 300})

acc = pd.read_pickle(f'{WORK}/acc.pkl'); bb = pd.read_pickle(f'{WORK}/bb.pkl')
ev = acc.groupby('event_id').agg(Mw=('Mw', 'first'), lat=('event_latitude_deg', 'first'), lon=('event_longitude_deg', 'first'))
pohang = ev[(ev.lat.between(35.9, 36.3)) & (ev.lon.between(129.2, 129.5))].index.tolist()
is_poh = acc.event_id.isin(pohang).values
is_poh_b = bb.event_id.isin(pohang).values
st = acc.groupby('station_code').agg(lat=('station_latitude_deg', 'first'), lon=('station_longitude_deg', 'first'))
st['bb'] = st.index.isin(bb.station_code.unique())
RED = '#a51d24'; BLUE = '#2f7fd1'

fig = plt.figure(figsize=(3.5, 7.3))
gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.30, left=0.14, right=0.97, top=0.957, bottom=0.065)
ax = fig.add_subplot(gs[0]); ax.grid(False)
m = Basemap(projection='merc', llcrnrlat=32.6, urcrnrlat=41.2, llcrnrlon=123.4, urcrnrlon=131.8, resolution='i', ax=ax)
m.drawcoastlines(linewidth=0.45, color='k'); m.drawcountries(linewidth=0.3, color='#555555')
m.fillcontinents(color='#f7f7f7', lake_color='white', zorder=0)
m.drawparallels(np.arange(34, 41, 3), labels=[1, 0, 0, 0], linewidth=0.3, color='#bbbbbb', fontsize=6.5, dashes=[1, 0])
m.drawmeridians(np.arange(124, 131, 3), labels=[0, 0, 1, 0], linewidth=0.3, color='#bbbbbb', fontsize=6.5, dashes=[1, 0])
for group, mk, col, lab in [((3.0, 4.0), 'o', '#8f8fd6', '$M_\\mathrm{W}$ 3–4'), ((4.0, 5.0), '^', '#2e7d32', '$M_\\mathrm{W}$ 4–5'), ((5.0, 6.0), 's', '#8a7a1e', '$M_\\mathrm{W}$ 5–6')]:
    k = (ev.Mw >= group[0]) & (ev.Mw < group[1]) & ~ev.index.isin(pohang)
    x, y = m(ev.lon[k].values, ev.lat[k].values)
    ax.scatter(x, y, marker=mk, s=22 if mk != 's' else 26, facecolors='none', edgecolors=col, lw=0.8, zorder=4, label=lab)
kp = ev.index.isin(pohang)
x, y = m(ev.lon[kp].values, ev.lat[kp].values)
ax.scatter(x, y, marker='o', s=18, facecolors='none', edgecolors='#777777', lw=0.6, zorder=4, label='Pohang sequence (comparison only)')
x, y = m(st.lon[~st.bb].values, st.lat[~st.bb].values)
ax.scatter(x, y, marker='v', s=16, color=RED, edgecolors='k', lw=0.3, zorder=3, label='ACC')
x, y = m(st.lon[st.bb].values, st.lat[st.bb].values)
ax.scatter(x, y, marker='v', s=16, color=BLUE, edgecolors='k', lw=0.3, zorder=3, label='ACC/BB')
h, l = ax.get_legend_handles_labels()
order = [4, 5, 0, 1, 2, 3]
ax.legend([h[i] for i in order], [l[i] for i in order], loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=6, handletextpad=0.3, borderpad=0.5, labelspacing=0.35, columnspacing=1.0, title='Stations / earthquakes', title_fontsize=6.2)
ax.text(-0.16, 1.045, '(a)', transform=ax.transAxes, fontsize=8, ha='left', va='bottom')

# (b) magnitude-distance distribution with marginal histograms
sub = gs[1].subgridspec(2, 2, width_ratios=[4.2, 1], height_ratios=[1, 3.6], wspace=0.06, hspace=0.08)
axm = fig.add_subplot(sub[1, 0]); axt = fig.add_subplot(sub[0, 0], sharex=axm); axr = fig.add_subplot(sub[1, 1], sharey=axm)
ka = ~is_poh; kb = ~is_poh_b
axm.scatter(acc.Rjb_km[ka], acc.Mw[ka], s=5, color=RED, lw=0, alpha=0.8, zorder=2, rasterized=True, label='ACC')
axm.scatter(bb.Rjb_km[kb], bb.Mw[kb], s=2.5, color=BLUE, lw=0, alpha=0.9, zorder=3, rasterized=True, label='BB')
axm.scatter(acc.Rjb_km[is_poh], acc.Mw[is_poh], s=4, facecolors='none', edgecolors='#777777', lw=0.3, zorder=4, rasterized=True, label='Pohang sequence (ACC)')
axm.set_xlim(0, 300); axm.set_ylim(2.85, 6.0); axm.set_xlabel('Joyner–Boore distance (km)'); axm.set_ylabel('Moment magnitude')
de = np.arange(0, 301, 25); me = np.arange(2.875, 6.01, 0.25)
axt.hist(acc.Rjb_km[ka], bins=de, color=RED, edgecolor='k', lw=0.3, label='ACC', zorder=2)
axt.hist(bb.Rjb_km[kb], bins=de, color=BLUE, edgecolor='k', lw=0.3, label='BB', zorder=3)
axt.hist(acc.Rjb_km[is_poh], bins=de, histtype='step', color='#555555', lw=0.7, label='Pohang sequence (ACC)', zorder=4)
axr.hist(acc.Mw[ka], bins=me, color=RED, edgecolor='k', lw=0.3, orientation='horizontal', zorder=2)
axr.hist(bb.Mw[kb], bins=me, color=BLUE, edgecolor='k', lw=0.3, orientation='horizontal', zorder=3)
axr.hist(acc.Mw[is_poh], bins=me, histtype='step', color='#555555', lw=0.7, orientation='horizontal', zorder=4)
axt.tick_params(labelbottom=False); axr.tick_params(labelleft=False)
axt.set_ylabel('Records', labelpad=2); axr.set_xlabel('Records', labelpad=2)
axt.set_yticks([0, 500, 1000]); axr.set_xticks([0, 2000]); axr.tick_params(axis='x', labelsize=5.5)
hh = [Patch(facecolor=RED, edgecolor='k', lw=0.3), Patch(facecolor=BLUE, edgecolor='k', lw=0.3), Line2D([0], [0], color='#555555', lw=0.7, marker='o', mfc='none', ms=3)]
axm.legend(hh, ['ACC', 'BB', 'Pohang sequence (ACC; comparison only)'], loc='center', bbox_to_anchor=(0.5, 0.80), ncol=3, fontsize=5.5, handlelength=1.2, handletextpad=0.4, borderpad=0.4, columnspacing=0.8)
axt.text(-0.13, 1.08, '(b)', transform=axt.transAxes, fontsize=8, ha='left', va='bottom')
fig.savefig(f'{FIG}/figure_1.pdf'); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
ax = axs[0]
ax.scatter(acc.Mw[~is_poh] + np.random.default_rng(1).uniform(-0.02, 0.02, (~is_poh).sum()), acc.highpass_hz[~is_poh], s=2, color='#bbbbbb', lw=0, rasterized=True, zorder=2, label='ACC records')
ax.scatter(bb.Mw[~is_poh_b] + np.random.default_rng(2).uniform(-0.02, 0.02, (~is_poh_b).sum()), bb.highpass_hz[~is_poh_b], s=2, color='#9ec7ee', lw=0, rasterized=True, zorder=3, label='BB records')
me2 = np.arange(2.75, 6.26, 0.5)
for data, mw, col, lab in [(acc.highpass_hz[~is_poh], acc.Mw[~is_poh], 'k', 'ACC, median per 0.5-magnitude bin'), (bb.highpass_hz[~is_poh_b], bb.Mw[~is_poh_b], BLUE, 'BB, median per 0.5-magnitude bin')]:
    cx, med = [], []
    for lo, hi in zip(me2[:-1], me2[1:]):
        k = (mw >= lo) & (mw < hi)
        if k.sum() >= 5:
            cx.append(float(np.mean(mw[k]))); med.append(np.median(data[k]))
    ax.plot(cx, med, '-o', color=col, ms=3, lw=1.0, zorder=5, label=lab)
ax.set_yscale('log'); ax.set_ylim(0.0012, 3); ax.set_xlim(2.85, 6.2)
ax.set_xlabel('Moment magnitude'); ax.set_ylabel('High-pass corner frequency (Hz)')
ax.legend(loc='lower right', fontsize=5.2, handlelength=1.3, handletextpad=0.4, borderpad=0.3, labelspacing=0.25, markerscale=1.2)
ax.text(-0.02, 1.02, '(a)', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)

ax = axs[1]
T = np.logspace(np.log10(0.03), np.log10(10), 120)
tmax = 1 / (FAC * acc.highpass_hz.values); tmin = FAC / acc.lowpass_hz.values
for (lo, hi), col in [((3.0, 3.5), '#8f8fd6'), ((3.5, 4.0), '#5a5ab8'), ((4.0, 4.5), '#2e7d32'), ((4.5, 6.0), '#c8963e')]:
    k = (~is_poh) & (acc.Mw.values >= lo) & (acc.Mw.values < hi)
    frac = [((t <= tmax[k]) & (t >= tmin[k])).mean() * 100 for t in T]
    ax.plot(T, frac, color=col, lw=1.1, label=f'$M_\\mathrm{{W}}$ {lo}–{hi if hi < 6 else 5.7} ({k.sum():,} records)')
ax.set_xscale('log'); ax.set_xlim(0.03, 10); ax.set_ylim(0, 102)
ax.axvspan(0.035, 3.0, color='#eeeeee', zorder=0, lw=0)
ax.set_xlabel('Period (s)'); ax.set_ylabel('Usable ACC records (%)')
ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.36), fontsize=5.2, handlelength=1.3, handletextpad=0.4, borderpad=0.3, labelspacing=0.25)
ax.text(-0.02, 1.02, '(b)', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)

ax = axs[2]
tmaxb = 1 / (FAC * bb.highpass_hz.values); tminb = FAC / bb.lowpass_hz.values
na = [((t <= tmax[~is_poh]) & (t >= tmin[~is_poh])).sum() for t in T]
nb = [((t <= tmaxb[~is_poh_b]) & (t >= tminb[~is_poh_b])).sum() for t in T]
evid = acc.event_id.values
nev = []
for t in T:
    k = (~is_poh) & (t <= tmax) & (t >= tmin)
    vc = pd.Series(evid[k]).value_counts(); nev.append(int((vc >= 3).sum()))
ax.plot(T, na, color='k', lw=1.2, label='ACC records')
ax.plot(T, nb, color=BLUE, lw=1.2, label='BB records')
ax.plot(T, nev, color='#c0272d', lw=1.2, label='Events with ≥ 3 usable ACC records')
ax.axvspan(0.035, 3.0, color='#eeeeee', zorder=0, lw=0)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.03, 10); ax.set_ylim(5, 2e4)
ax.set_xlabel('Period (s)'); ax.set_ylabel('Number')
ax.legend(loc='upper right', fontsize=5.2, handlelength=1.3, handletextpad=0.4, borderpad=0.3, labelspacing=0.25)
ax.text(-0.02, 1.02, '(c)', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
fig.tight_layout(w_pad=1.0)
fig.savefig(f'{FIG}/figure_S2.pdf'); plt.close(fig)
