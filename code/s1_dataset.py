# Moment magnitudes and residuals relative to BSSA14.
import re, logging, warnings
logging.disable(logging.WARNING); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import pygmm
from pygmm import BooreStewartSeyhanAtkinson2014 as B
from config import DATA, OUT, WORK, mw_from_ml


def residuals(df):
    sa = [c for c in df.columns if re.fullmatch(r'SA\([\d.]+\)_g', c)]
    T = np.array([float(re.search(r'SA\(([\d.]+)\)', c).group(1)) for c in sa])
    out = np.full((len(df), 2 + len(sa)), np.nan)
    idx = None
    for i, (m, r) in enumerate(zip(df.Mw.values, df.Rjb_km.values)):
        g = B(pygmm.Scenario(mag=float(m), dist_jb=float(r), v_s30=760, mechanism='SS', region='global'))
        per = np.asarray(g.periods); s = np.asarray(g.spec_accels)
        if idx is None:
            idx = np.array([np.argmin(abs(per - t)) for t in T])
            assert np.max(abs(per[idx] - T)) < 1e-6
        out[i, 0] = np.log(df.PGA_g.values[i]) - np.log(g.pga)
        out[i, 1] = np.log(df.PGV_cm_s.values[i]) - np.log(g.pgv)
        out[i, 2:] = np.log(df[sa].values[i].astype(float)) - np.log(s[idx])
    return pd.DataFrame(out, columns=['PGA_g', 'PGV_cm_s'] + sa, index=df.index)


if __name__ == '__main__':
    for sens, ff in (('acc', 'Flatfile_ACC.xlsx'), ('bb', 'Flatfile_BB.xlsx')):
        df = pd.read_excel(f'{DATA}/{ff}')
        df['Mw'] = [mw_from_ml(m, t) for m, t in zip(df.ML.values, df.origin_time_utc.values)]
        res = residuals(df)
        df.to_pickle(f'{WORK}/{sens}.pkl'); res.to_pickle(f'{WORK}/res_{sens}.pkl')
        print(sens, len(df), 'records', df.event_id.nunique(), 'events', df.station_code.nunique(), 'stations')
        if sens == 'acc':
            ev = df.groupby('event_id').agg(origin_time_utc=('origin_time_utc', 'first'), lat=('event_latitude_deg', 'first'),
                                            lon=('event_longitude_deg', 'first'), depth_km=('event_depth_km', 'first'),
                                            ML=('ML', 'first'), Mw=('Mw', 'first'), n_acc=('station_code', 'size'))
            ev['pohang_sequence'] = ev.lat.between(35.9, 36.3) & ev.lon.between(129.2, 129.5)
            ev.to_csv(f'{OUT}/events.csv')
            reg = ev[~ev.pohang_sequence]
            print('Pohang sequence', int(ev.pohang_sequence.sum()), 'events', int(ev.n_acc[ev.pohang_sequence].sum()), 'records')
            print('regression events', len(reg), 'records', int(reg.n_acc.sum()), 'ML %.1f-%.1f' % (reg.ML.min(), reg.ML.max()),
                  'MW %.3f-%.3f' % (reg.Mw.min(), reg.Mw.max()))
