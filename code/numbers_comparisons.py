# Residual statistics of the model comparisons quoted in the text (Figures 2-7, 14-18).
import pickle, logging, warnings
logging.disable(logging.WARNING); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import pygmm
from pygmm import BooreStewartSeyhanAtkinson2014 as B
from config import OUT, WORK, FAC, TOL, MCAP
from previous_gmms import EMOLO15, KHNP15, JH22

acc = pd.read_pickle(f'{WORK}/acc.pkl'); res = pd.read_pickle(f'{WORK}/res_acc.pkl')
sm = pd.read_pickle(f'{WORK}/coeff_main_smoothed.pkl').set_index('IM')
with open(f'{WORK}/final_effects.pkl', 'rb') as fh:
    final = pickle.load(fh)
ev = pd.read_csv(f'{OUT}/events.csv', index_col=0)
is_poh = acc.event_id.isin(ev.index[ev.pohang_sequence]).values
tmax = 1.0 / (FAC * acc.highpass_hz.values); tmin = FAC / acc.lowpass_hz.values
PER = {im: (np.nan if im in ('PGV_cm_s', 'PGA_g') else float(im[3:-3])) for im in sm.index}
_cache = {}


def usable(im):
    t = PER[im]
    return np.ones(len(acc), bool) if np.isnan(t) else ((t <= tmax + TOL) & (t >= tmin - TOL))


def ref_bssa(im, M, R):
    key = (im, round(float(M), 4), round(float(R), 3))
    if key not in _cache:
        mod = B(pygmm.Scenario(mag=float(M), dist_jb=float(R), v_s30=760, mechanism='SS', region='global'))
        if im == 'PGV_cm_s': v = mod.pgv
        elif im == 'PGA_g': v = mod.pga
        else:
            per = np.array(mod.periods); v = np.array(mod.spec_accels)[np.argmin(abs(per - PER[im]))]
        _cache[key] = float(v)
    return _cache[key]


def model_ln(im, M, R):
    r = sm.loc[im]
    ref = np.array([ref_bssa(im, m, x) for m, x in zip(M, R)])
    return np.log(ref) + r.a_s + r.a1_s * R + r.b1_s * np.minimum(MCAP, M)


def imt_name(im):
    return {'PGV_cm_s': 'PGV', 'PGA_g': 'PGA'}.get(im, im.replace('_g', ''))


def prev_models(im, sub):
    name = imt_name(im); Mw = sub.Mw.values.astype(float)
    ml = (1.086 * Mw - 0.4772 - 0.3262) / 0.9234     # ML on the KMA scale used before July 2019
    out = {'This study': np.exp(model_ln(im, Mw, sub.Rjb_km.values)),
           'BSSA14': np.array([ref_bssa(im, m, x) for m, x in zip(Mw, sub.Rjb_km.values)])}
    for key, fn in (('EMOLO15', lambda: EMOLO15(name, sub.Repi_km.values.astype(float), ml)._compute()),
                    ('KHNP15', lambda: KHNP15(name, sub.Rhypo_km.values.astype(float), Mw)._compute()),
                    ('JH22', lambda: JH22(name, sub.Rhypo_km.values.astype(float), Mw)._compute())):
        try:
            out[key] = np.asarray(fn())
        except KeyError:
            pass                      # no coefficients for this IM
    return out


def binned(x, y, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < hi)
        if k.sum():
            out.append((lo, hi, int(k.sum()), y[k].mean()))
    return out


def summary(label, im, sub):
    p = prev_models(im, sub); R = sub.Rjb_km.values; obs = sub[im].values
    print('  %s %s (%d records)' % (label, imt_name(im), len(sub)))
    for name, pr in p.items():
        r = np.log(obs) - np.log(pr); b = binned(R, r, np.arange(0, 301, 50))
        far = R >= 50
        print('    %-10s mean beyond 50 km %+.2f | bins beyond 50 km %s | all bins %s' % (
            name, r[far].mean() if far.any() else np.nan, ' '.join('%+.2f' % x[3] for x in b if x[0] >= 50), ' '.join('%d:%+.2f(%d)' % (x[0], x[3], x[2]) for x in b)))
    near = R < 50
    if near.any():
        print('    This study records within 50 km: %d, mean residual %+.2f' % (near.sum(), (np.log(obs) - np.log(p['This study']))[near].mean()))


main = acc[~is_poh]
print('Figure 14 (PGA)')
for M in (3.25, 4.25, 5.25):
    sub = main[(main.Mw >= M - 0.25) & (main.Mw <= M + 0.25)]
    summary('MW %.2f +-0.25, %d events,' % (M, sub.event_id.nunique()), 'PGA_g', sub)
for eq, lab in (('kr2019202020418', 'Sangju'), ('kr2016256113254', 'Gyeongju')):
    sub0 = acc[acc.event_id == eq]
    print('Figure %s: MW %.3f ML %.1f depth %.1f' % (lab, sub0.Mw.iloc[0], sub0.ML.iloc[0], sub0.event_depth_km.iloc[0]))
    for im in ('PGV_cm_s', 'SA(0.100)_g', 'SA(0.200)_g', 'SA(0.500)_g', 'SA(1.000)_g'):
        u = usable(im)[acc.index.get_indexer(sub0.index)]
        summary(lab, im, sub0[u])

print('Figures 2-3: residuals relative to BSSA14 (Pohang excluded, usable records, events with >= 3)')
for im in ('PGA_g', 'SA(0.050)_g', 'SA(0.100)_g', 'SA(2.000)_g', 'SA(3.000)_g'):
    m = final[im]['mask']; y = res[im].values
    bd = binned(acc.Rjb_km.values[m], y[m], np.arange(0, 301, 50)); bm = binned(acc.Mw.values[m], y[m], np.arange(3.0, 6.01, 0.5))
    print('  %-12s by distance %s | by magnitude %s' % (im, ' '.join('%+.2f' % x[3] for x in bd), ' '.join('%+.2f' % x[3] for x in bm)))
print('Figures 4-6: residuals relative to the model, within-event residuals')
for im in sm.index:
    if im not in ('PGV_cm_s', 'PGA_g', 'SA(0.050)_g', 'SA(0.100)_g', 'SA(0.300)_g', 'SA(1.000)_g', 'SA(2.000)_g', 'SA(3.000)_g'):
        continue
    m = final[im]['mask']; y = final[im]['corrected'].values
    bd = binned(acc.Rjb_km.values[m], y[m], np.arange(0, 301, 50)); bm = binned(acc.Mw.values[m], y[m], np.arange(3.0, 6.01, 0.5))
    eps = final[im]['eps']; bw = binned(acc.Rjb_km.values[acc.index.get_indexer(eps.index)], eps.values, np.arange(0, 301, 50))
    print('  %-12s total by R %s | by M %s | within-event by R %s' % (im, ' '.join('%+.2f' % x[3] for x in bd), ' '.join('%+.2f' % x[3] for x in bm), ' '.join('%+.2f' % x[3] for x in bw)))

with open(f'{WORK}/stoch_psa.pkl', 'rb') as fh:
    stoch = pickle.load(fh)
print('Figures 17-18: stochastic residuals (usable records, Pohang excluded, bins with >= 5 records)')
for im in ('PGV_cm_s', 'PGA_g', 'SA(0.050)_g', 'SA(0.100)_g', 'SA(0.300)_g', 'SA(1.000)_g', 'SA(2.000)_g', 'SA(3.000)_g'):
    m = usable(im) & ~is_poh
    for name, y in (('model', final[im]['corrected'].values), ('WNA', np.log(acc[im].values) - np.log(stoch['WNA'][im].values)),
                    ('SKP', np.log(acc[im].values) - np.log(stoch['SKP'][im].values))):
        bd = [x for x in binned(acc.Rjb_km.values[m], y[m], np.arange(0, 301, 50)) if x[2] >= 5]
        bm = [x for x in binned(acc.Mw.values[m], y[m], np.arange(3.0, 6.01, 0.5)) if x[2] >= 5]
        print('  %-12s %-5s by R %s | by M %s' % (im, name, ' '.join('%+.2f' % x[3] for x in bd), ' '.join('%+.2f' % x[3] for x in bm)))
