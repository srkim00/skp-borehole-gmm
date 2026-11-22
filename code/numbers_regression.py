# Numbers of the regressions quoted in the text.
import pickle
import pandas as pd
from config import OUT, WORK, MCAP
ev = pd.read_csv(f'{OUT}/events.csv', index_col=0)
main = pd.read_pickle(f'{WORK}/coeff_main.pkl').set_index('IM'); sm = pd.read_pickle(f'{WORK}/coeff_main_smoothed.pkl').set_index('IM')
po = pd.read_pickle(f'{WORK}/coeff_pohang_in.pkl').set_index('IM'); gy = pd.read_pickle(f'{WORK}/coeff_gy55.pkl').set_index('IM')
m4 = pd.read_pickle(f'{WORK}/coeff_m4.pkl').set_index('IM')
bbv = pd.read_pickle(f'{WORK}/bb_variability.pkl').set_index('IM')
sa = main['T'].notna()
T = main['T']
p = ev[ev.pohang_sequence]
print('Pohang sequence: %d events, MW %.2f-%.2f, %d records' % (len(p), p.Mw.min(), p.Mw.max(), p.n_acc.sum()))
print('Fig 8 / adopted: a1_s %.4f-%.4f per km (max at T=%s); b1_s %.3f to %.3f; se_b1 at T<=1 s %.3f-%.3f, at 3 s %.3f' % (
    sm.a1_s.min(), sm.a1_s.max(), sm.a1_s.idxmax(), sm.b1_s.max(), sm.b1_s.min(),
    main.se_b1[sa & (T <= 1.0)].min(), main.se_b1[sa & (T <= 1.0)].max(), main.se_b1['SA(3.000)_g']))
print('  b1_s at PGV %.3f PGA %.3f; a1_s>0 everywhere: %s' % (sm.b1_s['PGV_cm_s'], sm.b1_s['PGA_g'], bool((sm.a1_s > 0).all())))
print('  SA(0.1) a1_s*300 = %.2f' % (sm.a1_s['SA(0.100)_g'] * 300))
# Gyeongju MW 5.5
d = (gy.b1 - main.b1).abs()
print('Gyeongju MW 5.5: |change of b1| at T<=1 s (incl. PGV, PGA) %.3f-%.3f; at 3 s %.3f; events at 3 s %d' % (
    d[~sa | (T <= 1.0)].min(), d[~sa | (T <= 1.0)].max(), d['SA(3.000)_g'], main.n_events['SA(3.000)_g']))
# Pohang included
ra = (po.a1 / main.a1 - 1) * 100
print('Pohang included: a1 change %% at T<=1 s max %.1f (min %.1f); at 3 s %.1f; |b1 change| max %.3f' % (
    ra[~sa | (T <= 1.0)].max(), ra[~sa | (T <= 1.0)].min(), ra['SA(3.000)_g'], (po.b1 - main.b1).abs().max()))
# MW >= 4 only
print('MW>=4 only: events %d-%d, records at PGA %d; b1 %+.2f to %+.2f; se_b1 %.2f-%.2f; at 1 s %+.2f vs adopted-period %+.2f (smoothed %+.2f)' % (
    m4.n_events.min(), m4.n_events.max(), m4.n['PGA_g'], m4.b1.max(), m4.b1.min(), m4.se_b1.min(), m4.se_b1.max(),
    m4.b1['SA(1.000)_g'], main.b1['SA(1.000)_g'], sm.b1_s['SA(1.000)_g']))
dd = []
for im in ['SA(1.500)_g', 'SA(1.600)_g', 'SA(1.700)_g', 'SA(1.800)_g', 'SA(1.900)_g', 'SA(2.000)_g']:
    if im not in m4.index: continue
    for R in (20, 50, 100):
        for M in (5.5, 5.6, 5.7):
            fa = sm.a_s[im] + sm.a1_s[im] * R + sm.b1_s[im] * min(MCAP, M)
            fr = m4.a[im] + m4.a1[im] * R + m4.b1[im] * min(MCAP, M)
            dd.append(fr - fa)
print('restricted minus adopted at 1.5-2.0 s, RJB 20-100 km, MW 5.5-5.7: %.2f to %.2f' % (min(dd), max(dd)))
# BB level
print('BB level: sigma %.2f-%.2f; mean residual %+.2f to %+.2f' % (bbv.sigma.min(), bbv.sigma.max(), bbv.bias.min(), bbv.bias.max()))
# between-event residuals for MW >= 4.5 (Fig 7)
with open(f'{WORK}/final_effects.pkl', 'rb') as fh:
    fin = pickle.load(fh)
for im in ('SA(1.000)_g', 'SA(2.000)_g', 'SA(3.000)_g', 'PGA_g'):
    eta = fin[im]['eta']; mw = ev.Mw.reindex(eta.index)
    print('between-event %s: MW>=4.5 n=%d mean %+.2f; MW<4.5 mean %+.2f' % (im, int((mw >= 4.5).sum()), eta[mw >= 4.5].mean(), eta[mw < 4.5].mean()))
