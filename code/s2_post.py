# Smoothing, adopted variability, conversion factors, BB-level variability.
import logging, warnings, pickle
logging.disable(logging.WARNING); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from mixed import fit_crossed
from config import OUT, WORK, FAC, TOL, MCAP, MIN_REC
from cf_windows import KEEP, EXCL

acc = pd.read_pickle(f'{WORK}/acc.pkl'); bb = pd.read_pickle(f'{WORK}/bb.pkl')
res_acc = pd.read_pickle(f'{WORK}/res_acc.pkl'); res_bb = pd.read_pickle(f'{WORK}/res_bb.pkl')
tab = pd.read_pickle(f'{WORK}/coeff_main.pkl')
IMS = tab.IM.tolist(); TIM = tab['T'].values
ev = acc.groupby('event_id').agg(lat=('event_latitude_deg', 'first'), lon=('event_longitude_deg', 'first'))
pohang = ev[(ev.lat.between(35.9, 36.3)) & (ev.lon.between(129.2, 129.5))].index.tolist()


def smooth_tri(v, w=(1, 2, 3, 2, 1)):
    v = np.asarray(v, float); n = len(v); h = len(w) // 2; outv = np.zeros(n)
    for i in range(n):
        s = 0.0; ws = 0.0
        for k in range(-h, h + 1):
            j = i + k
            if 0 <= j < n:
                s += w[k + h] * v[j]; ws += w[k + h]
        outv[i] = s / ws
    return outv


def usable(t, hp, lp):
    if np.isnan(t):
        return np.ones(len(hp), bool)
    return (t <= 1.0 / (FAC * hp) + TOL) & (t >= FAC / lp - TOL)


if __name__ == '__main__':
    # smoothed coefficients (SA only)
    sm = tab.copy(); isa = tab['T'].notna().values
    for c in ['a', 'a1', 'b1']:
        sm.loc[isa, c + '_s'] = smooth_tri(tab.loc[isa, c].values); sm.loc[~isa, c + '_s'] = tab.loc[~isa, c].values
    for c in ['tau', 'phiS2S', 'phiSS']:
        sm.loc[isa, c + '_s'] = np.exp(smooth_tri(np.log(tab.loc[isa, c].values))); sm.loc[~isa, c + '_s'] = tab.loc[~isa, c].values
    sm['phi_s'] = np.sqrt(sm.phiS2S_s ** 2 + sm.phiSS_s ** 2); sm['sigma_s'] = np.sqrt(sm.tau_s ** 2 + sm.phi_s ** 2)

    X = np.column_stack([np.ones(len(acc)), acc.Rjb_km.values, np.minimum(MCAP, acc.Mw.values)])
    main_mask = ~acc.event_id.isin(pohang).values
    final = {}
    for im, t in zip(IMS, TIM):
        m = main_mask & usable(t, acc.highpass_hz.values, acc.lowpass_hz.values)
        cnt = acc.event_id[m].map(acc.event_id[m].value_counts()); m2 = m.copy(); m2[m] = (cnt >= MIN_REC).values
        row = sm[sm.IM == im].iloc[0]
        F = row.a_s + row.a1_s * X[:, 1] + row.b1_s * X[:, 2]
        y = res_acc[im].values - F
        f = fit_crossed(None, y[m2], acc.event_id.values[m2], acc.station_code.values[m2])
        final[im] = dict(mask=m2, corrected=pd.Series(y, index=acc.index), eta=pd.Series(f['eta'], index=f['events']),
                         dS=pd.Series(f['dS'], index=f['stations']), eps=pd.Series(f['eps'], index=acc.index[m2]),
                         tau=f['tau'], phiS2S=f['phiS2S'], phiSS=f['phiSS'], phi=f['phi'], sigma=f['sigma'])
        sm.loc[sm.IM == im, ['tau_f', 'phiS2S_f', 'phiSS_f', 'phi_f', 'sigma_f', 'n_var', 'n_events_var']] = \
            [f['tau'], f['phiS2S'], f['phiSS'], f['phi'], f['sigma'], int(m2.sum()), f['n_events']]
    with open(f'{WORK}/final_effects.pkl', 'wb') as fh:
        pickle.dump(final, fh)
    sm.to_pickle(f'{WORK}/coeff_main_smoothed.pkl'); sm.to_csv(f'{OUT}/coeff_main_smoothed.csv', index=False)

    # conversion factors (BB/ACC)
    pair = acc.merge(bb, on=['event_id', 'station_code'], suffixes=('_a', '_b'))
    pair['t'] = pd.to_datetime(pair.origin_time_utc_a, utc=True)
    keep = pd.Series(True, index=pair.index)
    for stn, fkeep in KEEP.items():
        mstn = pair.station_code == stn
        if mstn.any():
            keep[mstn] = fkeep(pair.t[mstn]).values
    p1 = pair[keep].copy()
    p1['tmax_pair'] = np.minimum(1 / (FAC * p1.highpass_hz_a.values), 1 / (FAC * p1.highpass_hz_b.values))
    p1['tmin_pair'] = np.maximum(FAC / p1.lowpass_hz_a.values, FAC / p1.lowpass_hz_b.values)
    depb = p1.groupby('station_code').sensor_depth_m_b.first()
    detail = {}

    def screen(p, im, m, lo=0.2, hi=5.0):
        r = (p[im + '_b'] / p[im + '_a'])[m]; sc = p.station_code[m]
        k0 = (r > lo) & (r < hi); out_range = r.index[~k0]
        lr = np.log(r[k0]); sc = sc[k0]
        st_sd = lr.groupby(sc).std()
        bad_st = st_sd[st_sd > st_sd.mean() + 3 * st_sd.std()].index
        k = ~sc.isin(bad_st); out_station = lr.index[~k]; lr = lr[k]; sc = sc[k]
        mu = lr.groupby(sc).transform('mean'); sd = lr.groupby(sc).transform('std')
        k2 = (lr - mu).abs() <= 3 * sd.fillna(np.inf)
        detail[im] = dict(range=list(out_range), station=list(out_station), record=list(lr.index[~k2]), bad_stations=list(bad_st))
        return lr[k2], sc[k2], int((~k0).sum()), len(bad_st), int((~k2).sum()), len(out_station)

    cf_rows = []; st_means = {}; st_sd = {}; st_n = {}
    for im, t in zip(IMS, TIM):
        m = np.ones(len(p1), bool) if np.isnan(t) else ((t <= p1.tmax_pair.values + TOL) & (t >= p1.tmin_pair.values - TOL))
        lr, sc, nr, nbs, nbr, nbs_rec = screen(p1, im, m)
        stm = lr.groupby(sc).mean(); st_means[im] = stm; st_sd[im] = lr.groupby(sc).std(); st_n[im] = lr.groupby(sc).size()
        g100 = stm[depb.reindex(stm.index).between(99, 106)]; g65 = stm[depb.reindex(stm.index).between(59, 68)]
        cf_rows.append(dict(IM=im, T=t, n_pairs=len(lr), n_stations=len(stm), CF=np.exp(stm.mean()), sd_ln_station=stm.std(),
                            CF_bb100=np.exp(g100.mean()), n_st_bb100=len(g100), CF_bb65=np.exp(g65.mean()), n_st_bb65=len(g65),
                            removed_range=nr, removed_stations=nbs, removed_station_records=nbs_rec, removed_records=nbr))
    cf = pd.DataFrame(cf_rows)
    cf.to_pickle(f'{WORK}/conv_factors.pkl'); cf.to_csv(f'{OUT}/conv_factors.csv', index=False)
    pd.DataFrame(st_means).to_pickle(f'{WORK}/station_mean_lnratio.pkl')
    pd.DataFrame(st_sd).to_pickle(f'{WORK}/station_sd_lnratio.pkl'); pd.DataFrame(st_n).to_pickle(f'{WORK}/station_n_pairs.pkl')
    p1.to_pickle(f'{WORK}/pairs_kept.pkl')
    with open(f'{WORK}/screen_detail.pkl', 'wb') as fh:
        pickle.dump(detail, fh)
    counts = dict(n_raw=len(pair), st_raw=int(pair.station_code.nunique()), n_win=int((~keep).sum()),
                  st_win=int(pair.station_code[~keep].nunique()), n_kept=len(p1), st_kept=int(p1.station_code.nunique()), EXCL=EXCL)
    with open(f'{WORK}/pair_screen_counts.pkl', 'wb') as fh:
        pickle.dump(counts, fh)
    print('pairs: raw', len(pair), 'at', counts['st_raw'], 'stations; removed by time intervals', counts['n_win'], 'at', counts['st_win'],
          'stations; kept', len(p1), 'at', counts['st_kept'], 'stations')

    # BB-level residuals relative to the converted model
    kept_keys = set(zip(p1.event_id, p1.station_code))
    bbk = bb[[k in kept_keys for k in zip(bb.event_id, bb.station_code)]].copy()
    bbk = bbk[~bbk.event_id.isin(pohang)]
    Xb = np.column_stack([np.ones(len(bbk)), bbk.Rjb_km.values, np.minimum(MCAP, bbk.Mw.values)])
    rows = []
    for im, t in zip(IMS, TIM):
        m = usable(t, bbk.highpass_hz.values, bbk.lowpass_hz.values)
        cnt = bbk.event_id[m].map(bbk.event_id[m].value_counts()); m2 = m.copy(); m2[m] = (cnt >= MIN_REC).values
        row = sm[sm.IM == im].iloc[0]; c = cf[cf.IM == im].iloc[0]
        F = row.a_s + row.a1_s * Xb[:, 1] + row.b1_s * Xb[:, 2]
        y = res_bb.loc[bbk.index, im].values - F - np.log(c.CF)
        f = fit_crossed(np.ones((int(m2.sum()), 1)), y[m2], bbk.event_id.values[m2], bbk.station_code.values[m2])
        rows.append(dict(IM=im, T=t, n=int(m2.sum()), n_events=f['n_events'], n_stations=f['n_stations'], bias=f['beta'][0], se_bias=f['se'][0],
                         tau=f['tau'], phiS2S=f['phiS2S'], phiSS=f['phiSS'], phi=f['phi'], sigma=f['sigma']))
    bbv = pd.DataFrame(rows); bbv.to_pickle(f'{WORK}/bb_variability.pkl'); bbv.to_csv(f'{OUT}/bb_variability.csv', index=False)
    print('post-processing done')
