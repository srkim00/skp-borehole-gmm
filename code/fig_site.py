# Figures S1 and S5.
import pickle, logging, warnings
logging.disable(logging.WARNING); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from config import WORK, DATA, FIG

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
                     'legend.fontsize': 6.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'axes.grid': True, 'grid.color': '#d9d9d9',
                     'grid.linestyle': '--', 'grid.linewidth': 0.4, 'legend.frameon': True, 'legend.framealpha': 1.0,
                     'legend.edgecolor': '#444444', 'pdf.fonttype': 42, 'savefig.dpi': 300})

meta = pd.read_excel(f'{DATA}/Site_metadata.xlsx').set_index('Station')
with open(f'{WORK}/final_effects.pkl', 'rb') as fh:
    final = pickle.load(fh)

H = meta.H_geo.where(meta.H_geo.notna(), meta.H_2020)      # engineering-bedrock depth: H_geo, else H_2020
hs = meta[H.notna()]; hv = H[H.notna()].values
fig = plt.figure(figsize=(3.5, 7.0))
ax = fig.add_axes([0.15, 0.475, 0.82, 0.475]); ax.grid(False)
m = Basemap(projection='merc', llcrnrlat=33.0, urcrnrlat=39.0, llcrnrlon=125.0, urcrnrlon=130.8, resolution='i', ax=ax)
m.drawcoastlines(linewidth=0.45, color='k'); m.drawcountries(linewidth=0.3, color='#555555')
m.drawparallels(np.arange(34, 39, 2), labels=[1, 0, 0, 0], linewidth=0.3, color='#bbbbbb', fontsize=6.5, dashes=[1, 0])
m.drawmeridians(np.arange(126, 131, 2), labels=[0, 0, 1, 0], linewidth=0.3, color='#bbbbbb', fontsize=6.5, dashes=[1, 0])
glon, glat = np.meshgrid(np.arange(125.0, 130.81, 0.04), np.arange(33.0, 39.01, 0.04))
gz = griddata(np.column_stack([hs.Longitude.values, hs.Latitude.values]), hv, (glon, glat), method='linear')
tree = cKDTree(np.column_stack([hs.Longitude.values, hs.Latitude.values]))
dist, _ = tree.query(np.column_stack([glon.ravel(), glat.ravel()])); dist = dist.reshape(glon.shape)
gx, gy = m(glon, glat)
land = np.array([m.is_land(x, y) for x, y in zip(gx.ravel(), gy.ravel())]).reshape(glon.shape)
gz = np.where(land & (dist < 0.35), gz, np.nan)
cmap = plt.get_cmap('RdBu_r'); norm = matplotlib.colors.Normalize(0, 50)
pm = ax.pcolormesh(gx, gy, gz, cmap=cmap, norm=norm, shading='auto', zorder=1, rasterized=True)
m.fillcontinents(color='none', lake_color='none', zorder=0)
sx, sy = m(meta.Longitude.values, meta.Latitude.values)
has_d = meta.Depth_ACC.notna().values
ax.scatter(sx[~has_d], sy[~has_d], marker='v', s=16, facecolors='white', edgecolors='#444444', lw=0.4, zorder=3)
ax.scatter(sx[has_d], sy[has_d], marker='v', s=16, c=meta.Depth_ACC.values[has_d], cmap=cmap, norm=norm, edgecolors='#444444', lw=0.4, zorder=3)
vs = meta.Vs_acc.notna().values
ax.scatter(sx[vs], sy[vs], marker='o', s=28, facecolors='none', edgecolors='#555555', lw=0.6, zorder=4)
cax = fig.add_axes([0.25, 0.425, 0.62, 0.013])
cb = fig.colorbar(pm, cax=cax, orientation='horizontal', extend='max'); cb.set_label('Depth (m): engineering bedrock (background), accelerometer sensor (triangles)', fontsize=6)
cb.ax.tick_params(labelsize=6)
ax.text(-0.16, 1.045, '(a)', transform=ax.transAxes, fontsize=8, ha='left', va='bottom')

# (b) histograms of VS at sensor depths
axb = fig.add_axes([0.15, 0.06, 0.82, 0.30])
bins = np.arange(250, 3501, 250)
va = meta.Vs_acc.dropna().values; vb = meta.Vs_bb.dropna().values
axb.hist([va, vb], bins=bins, stacked=True, color=['#f4a3a3', '#a9cfe8'], edgecolor='k', lw=0.4,
         label=[f'ACC ({len(va)} stations; mean $V_\\mathrm{{S}}$ = {va.mean():,.0f} m/s)', f'BB ({len(vb)} stations; mean $V_\\mathrm{{S}}$ = {vb.mean():,.0f} m/s)'])
axb.set_ylim(0, 27)
for v, col in [(va.mean(), '#c0272d'), (vb.mean(), '#1f3fa8')]:
    axb.axvline(v, ymax=0.82, color=col, ls='--', lw=1.2)
axb.set_xlabel('Shear-wave velocity at the sensor depth (m/s)'); axb.set_ylabel('Number of stations')
axb.set_xlim(250, 3500); axb.set_yticks([0, 5, 10, 15, 20, 25]); axb.legend(loc='upper left', fontsize=6, ncol=1)
axb.text(-0.16, 1.02, '(b)', transform=axb.transAxes, fontsize=8, ha='left', va='bottom')
fig.savefig(f'{FIG}/figure_S1.pdf'); plt.close(fig)

IMS = ['SA(0.050)_g', 'SA(0.100)_g', 'SA(0.300)_g', 'SA(1.000)_g']
rows = [('Depth_ACC', 'Accelerometer sensor depth (m)', (8, 500), [10, 20, 50, 100, 200, 500]),
        ('Vs30', '$V_\\mathrm{S30}$ (m/s)', (200, 2000), [200, 300, 500, 1000, 2000]),
        ('Vs_acc', '$V_\\mathrm{S}$ at the sensor depth (m/s)', (300, 3500), [300, 500, 1000, 3000])]
fig, axs = plt.subplots(3, 4, figsize=(7.0, 5.6), sharey=True)
k = 0
for ri, (col, lab, xl, tk) in enumerate(rows):
    for ci, im in enumerate(IMS):
        ax = axs[ri, ci]; dS = final[im]['dS']; x = meta[col].reindex(dS.index); ok = x.notna()
        ax.scatter(x[ok], dS[ok], s=6, color='#1f3fa8', lw=0, alpha=0.8, zorder=2)
        r = pearsonr(np.log10(x[ok]), dS[ok])[0]
        ax.axhline(0, color='k', lw=0.6, zorder=1); ax.set_xscale('log'); ax.set_xlim(*xl); ax.set_ylim(-1.5, 1.8)
        ax.set_xticks(tk); ax.set_xticklabels([str(t) for t in tk]); ax.set_xticks([], minor=True)
        ax.text(0.97, 0.05, im.replace('_g', ''), transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        rs = f'{abs(r):.2f}' if abs(r) < 0.005 else (('−' if r < 0 else '+') + f'{abs(r):.2f}')
        ax.set_title(f'$r$ = {rs} ({int(ok.sum())} stations)', fontsize=6, loc='right', pad=3)
        ax.text(-0.02, 1.02, f'({"abcdefghijkl"[k]})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8); k += 1
        ax.set_xlabel(lab)
    axs[ri, 0].set_ylabel('Station term (ln)')
fig.tight_layout(h_pad=1.0)
fig.savefig(f'{FIG}/figure_S5.pdf'); plt.close(fig)
