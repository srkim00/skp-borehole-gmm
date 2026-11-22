# Quadratic and bilinear magnitude terms: fit and withheld events with MW >= 4.
import numpy as np, pandas as pd
from scipy.stats import chi2
from withheld_lib import *
from s3_withheld import folds

IMSEL = ['PGV_cm_s', 'PGA_g', 'SA(0.050)_g', 'SA(0.100)_g', 'SA(0.200)_g', 'SA(0.300)_g', 'SA(0.500)_g', 'SA(0.750)_g',
         'SA(1.000)_g', 'SA(1.500)_g', 'SA(2.000)_g', 'SA(3.000)_g']
Mc = np.minimum(MCAP, acc.Mw.values)
X = {'L': XFULL, 'Q': np.column_stack([XFULL, (Mc - 4.0) ** 2]), 'B': np.column_stack([XFULL, np.maximum(Mc - 4.0, 0.0)])}


def fit(mask, im, name):
    return fit_crossed(X[name][mask], res[im].values[mask], acc.event_id.values[mask], acc.station_code.values[mask])


if __name__ == '__main__':
    key = acc.event_id.values
    fo = folds(np.unique(key[MAIN])); fold = np.array([fo.get(x, -1) for x in key])
    rows = []
    for im in IMSEL:
        tp = test_population(im)
        f = {n: fit(tp, im, n) for n in X}
        row = dict(IM=im, records=int(tp.sum()), events=f['L']['n_events'])
        for n in ('Q', 'B'):
            lr = 2 * (f['L']['nll'] - f[n]['nll'])
            row.update({f'LR_{n}': lr, f'p_{n}': chi2.sf(max(lr, 0), 1), f'b2_{n}': f[n]['beta'][3], f'se_b2_{n}': f[n]['se'][3],
                        f'tau_{n}': f[n]['tau']})
        row['tau_L'] = f['L']['tau']
        pred = {n: np.full(len(acc), np.nan) for n in X}
        for k in range(5):
            tr = min_rec(MAIN & (fold != k) & usable(im)); te = tp & (fold == k)
            for n in X:
                b = fit(tr, im, n)['beta']; pred[n][te] = X[n][te] @ b
        y = res[im].values; big = tp & (acc.Mw.values >= 4.0)
        for n in X:
            row[f'RMSE4_{n}'] = event_rmse((y - pred[n])[big], key[big])[0]
            row[f'RMSEall_{n}'] = event_rmse((y - pred[n])[tp], key[tp])[0]
        row['events4'] = len(np.unique(key[big]))
        rows.append(row)
        print(im, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != 'IM'}, flush=True)
    d = pd.DataFrame(rows); d.to_csv(f'{OUT}/magnitude_terms.csv', index=False)
    pd.set_option('display.width', 250)
    print(d[['IM', 'events', 'LR_Q', 'p_Q', 'b2_Q', 'se_b2_Q', 'LR_B', 'p_B', 'b2_B', 'se_b2_B', 'tau_L', 'tau_Q', 'tau_B']].round(4).to_string(index=False))
    print(d[['IM', 'events4', 'RMSE4_L', 'RMSE4_Q', 'RMSE4_B', 'RMSEall_L', 'RMSEall_Q', 'RMSEall_B']].round(4).to_string(index=False))
