# Shared set-up of the withholding tests.
import numpy as np, pandas as pd
from mixed import fit_crossed
from config import OUT, WORK, FAC, TOL, MCAP, MIN_REC

acc = pd.read_pickle(f'{WORK}/acc.pkl')
res = pd.read_pickle(f'{WORK}/res_acc.pkl')
tab = pd.read_pickle(f'{WORK}/coeff_main.pkl')
IMS = tab.IM.tolist(); TIM = dict(zip(tab.IM, tab['T']))
SA = [im for im in IMS if im.startswith('SA(')]
ev = acc.groupby('event_id').agg(lat=('event_latitude_deg', 'first'), lon=('event_longitude_deg', 'first'))
POHANG = ev[(ev.lat.between(35.9, 36.3)) & (ev.lon.between(129.2, 129.5))].index
MAIN = ~acc.event_id.isin(POHANG).values
TMAX = 1.0 / (FAC * acc.highpass_hz.values); TMIN = FAC / acc.lowpass_hz.values
EVAL = ['PGV_cm_s', 'PGA_g', 'SA(0.050)_g', 'SA(0.100)_g', 'SA(0.300)_g', 'SA(1.000)_g', 'SA(2.000)_g', 'SA(3.000)_g']
XFULL = np.column_stack([np.ones(len(acc)), acc.Rjb_km.values, np.minimum(MCAP, acc.Mw.values)])
MODELS = {'const': [0], 'dist': [0, 1], 'rem': [0, 1, 2]}


def usable(im):
    t = TIM[im]
    if np.isnan(t):
        return np.ones(len(acc), bool)
    return (t <= TMAX + TOL) & (t >= TMIN - TOL)


def min_rec(mask):
    cnt = acc.event_id[mask].map(acc.event_id[mask].value_counts())
    m2 = mask.copy(); m2[mask] = (cnt >= MIN_REC).values
    return m2


def test_population(im):
    return min_rec(MAIN & usable(im))


def window(im):
    i = SA.index(im); w = (1, 2, 3, 2, 1)
    return [(SA[j], w[k]) for k, j in enumerate(range(i - 2, i + 3)) if 0 <= j < len(SA)]


def fit_models(train, im, models=('const', 'dist', 'rem')):
    m = min_rec(train & usable(im))
    out = {}
    for name in models:
        cols = MODELS[name]
        f = fit_crossed(XFULL[m][:, cols], res[im].values[m], acc.event_id.values[m], acc.station_code.values[m])
        out[name] = f['beta']
    return out


def event_rmse(resid, events):
    s = pd.Series(resid ** 2).groupby(np.asarray(events)).mean()
    return float(np.sqrt(s.mean())), int(len(s))


def event_mean(resid, events):
    s = pd.Series(resid).groupby(np.asarray(events)).mean()
    return float(s.mean())
