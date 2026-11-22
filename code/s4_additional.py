# Station-level predictors and leave-one-event-out b1 (Table S7).
import os, time
import numpy as np, pandas as pd
from scipy.stats import chi2
import s2_fit as S
from mixed import fit_crossed
from config import OUT, DATA

acc = S.acc
site = pd.read_excel(os.path.join(DATA, 'Site_metadata.xlsx')).set_index('Station')
st = acc.groupby('station_code').agg(h=('sensor_depth_m', 'first')).join(site[['Vs30', 'Vs_acc']])
H0, V0, VA0 = 23.0, 600.0, 1900.0
PER = [0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
IMS = ['PGV_cm_s', 'PGA_g'] + [im for im, t in zip(S.IMS, S.TIM) if not np.isnan(t) and any(abs(t - x) < 1e-6 for x in PER)]
TIM = dict(zip(S.IMS, S.TIM))
TERMS = {'ln(h/23 m)': lambda s: np.log(s.h.values / H0), 'ln(VS30/600 m/s)': lambda s: np.log(s.Vs30.values / V0),
         'ln(VS_acc/1900 m/s)': lambda s: np.log(s.Vs_acc.values / VA0)}


def station_predictors():
    ok = set(st.index[st.h.notna() & st.Vs30.notna() & st.Vs_acc.notna()])
    r = S.residuals_for('main'); X = S.Xmat('main'); rows = []
    for im in IMS:
        t = TIM[im]
        m = S.dataset_mask('main') & S.usable_mask(t) & acc.station_code.isin(ok).values
        cnt = acc.event_id[m].map(acc.event_id[m].value_counts()); m2 = m.copy(); m2[m] = (cnt >= S.MIN_REC).values
        y = r[im].values[m2]; X0 = X[m2]; ev = acc.event_id.values[m2]; sc = acc.station_code.values[m2]
        f0 = fit_crossed(X0, y, ev, sc)
        rows.append(dict(IM=im, T=t, model='base (equation 3)', n=int(m2.sum()), n_events=f0['n_events'], n_stations=f0['n_stations'],
                         a=f0['beta'][0], a1=f0['beta'][1], b1=f0['beta'][2], tau=f0['tau'], phiS2S=f0['phiS2S'], phiSS=f0['phiSS'],
                         sigma=f0['sigma'], nll=f0['nll']))
        s = st.loc[sc]
        for name, fn in TERMS.items():
            fk = fit_crossed(np.column_stack([X0, fn(s)]), y, ev, sc)
            lr = 2 * (f0['nll'] - fk['nll'])
            rows.append(dict(IM=im, T=t, model='+ ' + name, n=int(m2.sum()), n_events=fk['n_events'], n_stations=fk['n_stations'],
                             a=fk['beta'][0], a1=fk['beta'][1], b1=fk['beta'][2], c=fk['beta'][3], se_c=fk['se'][3],
                             tau=fk['tau'], phiS2S=fk['phiS2S'], phiSS=fk['phiSS'], sigma=fk['sigma'], nll=fk['nll'], LR=lr,
                             p=chi2.sf(lr, 1), dphiS2S_pct=100 * (fk['phiS2S'] / f0['phiS2S'] - 1)))
        print(im, int(m2.sum()), f0['n_stations'], flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(f'{OUT}/station_predictor_tests.csv', index=False)
    return tab


def jackknife_m4():
    r = S.residuals_for('m4'); X = S.Xmat('m4'); rows = []
    for im, t in zip(S.IMS, S.TIM):
        if np.isnan(t) or t < 1.5 - 1e-9:
            continue
        m2 = S.records_mask('m4', t)
        evs = pd.unique(acc.event_id.values[m2])
        f0 = fit_crossed(X[m2], r[im].values[m2], acc.event_id.values[m2], acc.station_code.values[m2])
        for e in evs:
            k = m2 & (acc.event_id.values != e)
            cnt = acc.event_id[k].map(acc.event_id[k].value_counts()); k2 = k.copy(); k2[k] = (cnt >= S.MIN_REC).values
            f = fit_crossed(X[k2], r[im].values[k2], acc.event_id.values[k2], acc.station_code.values[k2])
            rows.append(dict(IM=im, T=t, n_events=len(evs), left_out=e, b1_full=f0['beta'][2], b1_loo=f['beta'][2], db1=f['beta'][2] - f0['beta'][2]))
    J = pd.DataFrame(rows); J.to_csv(f'{OUT}/jackknife_b1_m4.csv', index=False)
    return J


if __name__ == '__main__':
    t0 = time.time()
    tab = station_predictors()
    J = jackknife_m4()
    print('max |db1| (MW >= 4, 1.5-3.0 s): %.3f' % J.db1.abs().max(), 'events per period %d-%d' % (J.n_events.min(), J.n_events.max()))
    print('done', round(time.time() - t0), 's')
