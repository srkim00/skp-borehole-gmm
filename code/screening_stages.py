# Conversion factor at each screening stage (Figure S3).
import numpy as np, pandas as pd
from config import OUT, WORK, FAC, TOL
from cf_windows import KEEP
acc = pd.read_pickle(f'{WORK}/acc.pkl'); bb = pd.read_pickle(f'{WORK}/bb.pkl')
cf = pd.read_pickle(f'{WORK}/conv_factors.pkl').set_index('IM')
pair = acc.merge(bb, on=['event_id', 'station_code'], suffixes=('_a', '_b'))
pair['t'] = pd.to_datetime(pair.origin_time_utc_a, utc=True)
keep = pd.Series(True, index=pair.index)
for stn, f in KEEP.items():
    m = pair.station_code == stn
    if m.any(): keep[m] = f(pair.t[m]).values
pair['tmax'] = np.minimum(1 / (FAC * pair.highpass_hz_a.values), 1 / (FAC * pair.highpass_hz_b.values))
pair['tmin'] = np.maximum(FAC / pair.lowpass_hz_a.values, FAC / pair.lowpass_hz_b.values)
rows = []
for im, t in zip(cf.index, cf['T']):
    out = dict(IM=im, T=t)
    for lab, sel in (('all', np.ones(len(pair), bool)), ('time', keep.values)):
        m = sel & (np.ones(len(pair), bool) if np.isnan(t) else ((t <= pair.tmax.values + TOL) & (t >= pair.tmin.values - TOL)))
        lr = np.log(pair[im + '_b'][m] / pair[im + '_a'][m])
        stm = lr.groupby(pair.station_code[m]).mean()
        out['CF_' + lab] = np.exp(stm.mean()); out['n_' + lab] = int(m.sum()); out['st_' + lab] = len(stm)
    out['CF_final'] = cf.CF[im]
    rows.append(out)
d = pd.DataFrame(rows); d.to_csv(f'{OUT}/screening_stages.csv', index=False)
print('PGA/PGV pair counts before period screening: all %d pairs at %d stations; after time exclusions %d at %d' % (
    d.n_all[1], d.st_all[1], d.n_time[1], d.st_time[1]))
dt = d.CF_all - d.CF_time; ds = d.CF_final - d.CF_time
print('time exclusions lowered CF by %.3f-%.3f; ratio screening changed CF by %.3f to %+.3f (max |%.3f|)' % (dt.min(), dt.max(), ds.min(), ds.max(), ds.abs().max()))
