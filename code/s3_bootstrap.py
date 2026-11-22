# Bootstrap intervals of the RMSE differences (Table S5).
import pickle
import numpy as np, pandas as pd
from withheld_lib import *
with open(f'{WORK}/withheld_predictions.pkl', 'rb') as fh:
    P = pickle.load(fh)
pe, ps = P['event'], P['station']
NB = 2000


def ev_boot(r1, r0, ev, rng):
    d = pd.DataFrame(dict(a=r1 ** 2, b=r0 ** 2, ev=ev)).groupby('ev').mean()
    a, b = d.a.values, d.b.values; n = len(a)
    idx = rng.integers(0, n, size=(NB, n))
    diff = np.sqrt(a[idx].mean(axis=1)) - np.sqrt(b[idx].mean(axis=1))
    return np.percentile(diff, [2.5, 97.5])


def st_boot(r1, r0, ev, st, rng):
    d = pd.DataFrame(dict(a=r1 ** 2, b=r0 ** 2, ev=ev, st=st, n=1)).groupby(['ev', 'st']).sum()
    evs = d.index.get_level_values(0); sts = d.index.get_level_values(1)
    su = np.unique(sts); s_i = pd.Index(su).get_indexer(sts); eu, e_i = np.unique(evs, return_inverse=True)
    out = np.empty(NB)
    for k in range(NB):
        w = np.bincount(rng.integers(0, len(su), len(su)), minlength=len(su))[s_i]
        A = np.bincount(e_i, weights=w * d.a.values, minlength=len(eu)); Bv = np.bincount(e_i, weights=w * d.b.values, minlength=len(eu))
        N = np.bincount(e_i, weights=w * d.n.values, minlength=len(eu)); ok = N > 0
        out[k] = np.sqrt((A[ok] / N[ok]).mean()) - np.sqrt((Bv[ok] / N[ok]).mean())
    return np.percentile(out, [2.5, 97.5])


rows = []
for im in EVAL:
    tp = test_population(im); ev = acc.event_id.values[tp]; y = res[im].values[tp]
    rem = y - pe[('rem', im)][tp]; dist = y - pe[('dist', im)][tp]; rems = y - ps[('rem', im)][tp]
    a = ev_boot(rem, y, ev, np.random.default_rng(20261024))
    b = ev_boot(rem, dist, ev, np.random.default_rng(20261024))
    c = st_boot(rems, y, ev, acc.station_code.values[tp], np.random.default_rng(20261024))
    rows.append(dict(IM=im, event_REM_minus_BSSA14_lo=a[0], event_REM_minus_BSSA14_hi=a[1], event_REM_minus_distance_lo=b[0],
                     event_REM_minus_distance_hi=b[1], station_REM_minus_BSSA14_lo=c[0], station_REM_minus_BSSA14_hi=c[1]))
    print('%-12s REM-BSSA14 (%+.3f, %+.3f)  REM-dist (%+.3f, %+.3f)  station REM-BSSA14 (%+.3f, %+.3f)' % (im, a[0], a[1], b[0], b[1], c[0], c[1]))
pd.DataFrame(rows).to_csv(f'{OUT}/withheld_bootstrap.csv', index=False)
