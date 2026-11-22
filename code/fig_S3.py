# Figure S3.
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import OUT, FIG

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'axes.labelsize': 7.5, 'legend.fontsize': 5.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.5,
                     'ytick.major.width': 0.5, 'axes.grid': True, 'grid.color': '#d9d9d9', 'grid.linestyle': '--',
                     'grid.linewidth': 0.4, 'legend.frameon': True, 'legend.framealpha': 1.0, 'legend.edgecolor': '#444444',
                     'pdf.fonttype': 42, 'savefig.dpi': 300})
XPGV, XPGA, XSEP = 0.0155, 0.0265, 0.0315
STAGES = [('CF_all', 'All available pairs', dict(color='#1f3fa8', lw=1.0, ls='-')),
          ('CF_time', 'Time-window exclusions only', dict(color='#c0272d', lw=1.0, ls='--')),
          ('CF_final', 'Final ratio screening', dict(color='k', lw=1.4, ls='-'))]


def draw(d):
    sa = d[d['T'].notna()]; pv = d[d.IM == 'PGV_cm_s'].iloc[0]; pa = d[d.IM == 'PGA_g'].iloc[0]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for col, lab, st in STAGES:
        ax.plot(sa['T'], sa[col], label=lab, **st)
        ax.plot([XPGV, XPGA], [pv[col], pa[col]], 'o', color=st['color'], ms=3, mew=0.4)
    ax.axhline(1.0, color='#777777', lw=0.6)
    ax.axvline(XSEP, color='#777777', lw=0.5, ls=':')
    ax.set_xscale('log'); ax.set_xlim(0.0135, 3.3); ax.set_ylim(0.3, 1.2)
    ax.set_xticks([XPGV, XPGA, 0.05, 0.1, 0.2, 0.5, 1, 2, 3])
    ax.set_xticklabels(['PGV', 'PGA', '0.05', '0.1', '0.2', '0.5', '1', '2', '3'])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(np.round(np.arange(0.3, 1.21, 0.1), 1))
    ax.set_xlabel('Period (s)'); ax.set_ylabel('Conversion factor (broadband / accelerometer)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    fig = draw(pd.read_csv(f'{OUT}/screening_stages.csv'))
    fig.savefig(f'{FIG}/figure_S3.pdf')
