# Regressions of equation (3): main, pohang_in, gy55, m4.
import sys, logging, warnings, re, time, pickle
logging.disable(logging.WARNING); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import pygmm
from pygmm import BooreStewartSeyhanAtkinson2014 as B
from mixed import fit_crossed
from config import OUT, WORK, FAC, TOL, MCAP, MIN_REC, GY

acc = pd.read_pickle(f'{WORK}/acc.pkl')
res = pd.read_pickle(f'{WORK}/res_acc.pkl')
sa_cols = [c for c in acc.columns if c.startswith('SA(')]
Tall = np.array([float(re.search(r'SA\(([\d.]+)\)', c).group(1)) for c in sa_cols])
keep = [c for c, t in zip(sa_cols, Tall) if 0.035 - 1e-9 <= t <= 3.0 + 1e-9]
IMS = ['PGV_cm_s', 'PGA_g'] + keep
TIM = [np.nan, np.nan] + [float(re.search(r'SA\(([\d.]+)\)', c).group(1)) for c in keep]
ev = acc.groupby('event_id').agg(lat=('event_latitude_deg', 'first'), lon=('event_longitude_deg', 'first'))
pohang = ev[(ev.lat.between(35.9, 36.3)) & (ev.lon.between(129.2, 129.5))].index.tolist()
tmax = 1.0 / (FAC * acc.highpass_hz.values)
tmin = FAC / acc.lowpass_hz.values


def usable_mask(t):
    if np.isnan(t):
        return np.ones(len(acc), bool)
    return (t <= tmax + TOL) & (t >= tmin - TOL)


def dataset_mask(name):
    m = np.ones(len(acc), bool)
    if name in ('main', 'gy55', 'm4'):
        m &= ~acc.event_id.isin(pohang).values
    if name == 'm4':
        m &= acc.Mw.values >= 4.0
    return m


def residuals_for(name):
    r = res[IMS].copy()
    if name == 'gy55':
        sub = acc[acc.event_id == GY]
        for i, R in zip(sub.index, sub.Rjb_km.values):
            m = B(pygmm.Scenario(mag=5.5, dist_jb=R, v_s30=760, mechanism='SS', region='global'))
            per = np.array(m.periods); sa = np.array(m.spec_accels)
            r.loc[i, 'PGA_g'] = np.log(acc.loc[i, 'PGA_g']) - np.log(m.pga)
            r.loc[i, 'PGV_cm_s'] = np.log(acc.loc[i, 'PGV_cm_s']) - np.log(m.pgv)
            for c, t in zip(keep, TIM[2:]):
                r.loc[i, c] = np.log(acc.loc[i, c]) - np.log(sa[np.argmin(abs(per - t))])
    return r


def magnitudes(name):
    M = acc.Mw.values.copy()
    if name == 'gy55':
        M[acc.event_id.values == GY] = 5.5
    return M


def Xmat(name):
    return np.column_stack([np.ones(len(acc)), acc.Rjb_km.values, np.minimum(MCAP, magnitudes(name))])


def records_mask(name, t):
    m = dataset_mask(name) & usable_mask(t)
    cnt = acc.event_id[m].map(acc.event_id[m].value_counts())
    m2 = m.copy(); m2[m] = (cnt >= MIN_REC).values
    return m2


def run(name):
    t0 = time.time()
    r = residuals_for(name); X = Xmat(name); M = magnitudes(name)
    rows = []; eff = {}
    for im, t in zip(IMS, TIM):
        m2 = records_mask(name, t)
        y = r[im].values[m2]
        f = fit_crossed(X[m2], y, acc.event_id.values[m2], acc.station_code.values[m2])
        rows.append(dict(IM=im, T=t, n=int(m2.sum()), n_events=f['n_events'], n_stations=f['n_stations'],
                         Mw_min=float(M[m2].min()), Mw_max=float(M[m2].max()),
                         a=f['beta'][0], a1=f['beta'][1], b1=f['beta'][2], se_a=f['se'][0], se_a1=f['se'][1], se_b1=f['se'][2],
                         tau=f['tau'], phiS2S=f['phiS2S'], phiSS=f['phiSS'], phi=f['phi'], sigma=f['sigma'], converged=f['converged']))
        if name == 'main':
            eff[im] = dict(mask=m2, eta=pd.Series(f['eta'], index=f['events']), dS=pd.Series(f['dS'], index=f['stations']),
                           eps=pd.Series(f['eps'], index=acc.index[m2]))
        print(name, im, int(m2.sum()), f['n_events'], np.round(f['beta'], 4), round(f['tau'], 3), round(f['phi'], 3), flush=True)
    tab = pd.DataFrame(rows)
    tab.to_pickle(f'{WORK}/coeff_{name}.pkl'); tab.to_csv(f'{OUT}/coeff_{name}.csv', index=False)
    if name == 'main':
        with open(f'{WORK}/effects_main.pkl', 'wb') as fh:
            pickle.dump(eff, fh)
    print(name, 'done in', round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    for nm in (sys.argv[1:] or ['main', 'pohang_in', 'gy55', 'm4']):
        run(nm)
