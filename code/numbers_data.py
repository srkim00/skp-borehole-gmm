# Numbers of the dataset, residuals, and conversion factors quoted in the text.
import os, pickle
import numpy as np, pandas as pd
from config import OUT, WORK, DATA
acc = pd.read_pickle(f'{WORK}/acc.pkl'); bb = pd.read_pickle(f'{WORK}/bb.pkl')
ev = pd.read_csv(f'{OUT}/events.csv', index_col=0)
main = pd.read_pickle(f'{WORK}/coeff_main.pkl').set_index('IM'); sm = pd.read_pickle(f'{WORK}/coeff_main_smoothed.pkl').set_index('IM')
cf = pd.read_pickle(f'{WORK}/conv_factors.pkl').set_index('IM')
with open(f'{WORK}/final_effects.pkl', 'rb') as fh:
    fin = pickle.load(fh)
po = acc.event_id.isin(ev.index[ev.pohang_sequence]).values
reg = ~po
site = pd.read_excel(os.path.join(DATA, 'Site_metadata.xlsx')).set_index('Station')
st = acc.groupby('station_code').agg(h=('sensor_depth_m', 'first'))
print('low-pass max %.1f Hz; fraction < 25 Hz: all %.3f, regression %.3f' % (acc.lowpass_hz.max(), (acc.lowpass_hz < 25).mean(), (acc.lowpass_hz[reg] < 25).mean()))
for lab, mm in (('MW<3.5', acc.Mw < 3.5), ('MW>=4.5', acc.Mw >= 4.5)):
    print('median high-pass %s: all %.3f, regression %.3f Hz' % (lab, acc.highpass_hz[mm].median(), acc.highpass_hz[mm & reg].median()))
sa = main['T'].notna()
print('records max %d at T=%s; 0.035 s %d; 3.0 s %d; events max %d, at 3 s %d' % (main.n.max(), list(main['T'][main.n == main.n.max()].round(3)),
      main.n['SA(0.035)_g'], main.n['SA(3.000)_g'], main.n_events.max(), main.n_events['SA(3.000)_g']))
print('events=100 at T:', list(main['T'][sa & (main.n_events == 100)].round(3)), '; at 2 s %d, 3 s %d' % (main.n_events['SA(2.000)_g'], main.n_events['SA(3.000)_g']))
for im in ('SA(2.000)_g', 'SA(3.000)_g'):
    m = fin[im]['mask']; e = pd.unique(acc.event_id[m]); print('%s events with MW>=4: %d of %d' % (im, int((ev.Mw.reindex(e) >= 4).sum()), len(e)))
# sensor depths and VS
sb = bb.groupby('station_code').sensor_depth_m.first()
print('mean ACC sensor depth %.1f m (n=%d); mean BB depth %.1f m (n=%d)' % (st.h.mean(), st.h.notna().sum(), sb.mean(), sb.notna().sum()))
s2 = site.reindex(st.index)
print('mean VS at ACC depth %.0f (n=%d), at BB depth %.0f (n=%d); ACC VS<760: %d' % (
    s2.Vs_acc.mean(), s2.Vs_acc.notna().sum(), site.reindex(sb.index).Vs_bb.mean(), site.reindex(sb.index).Vs_bb.notna().sum(),
    int((s2.Vs_acc < 760).sum())))
print('ACC depths 20-30 m: %d of %d' % (int(st.h.between(20, 30).sum()), st.h.notna().sum()))
# variance components (adopted, _f)
s = sm[sa]
print('sigma %.2f-%.2f (all IMs %.2f-%.2f); tau %.2f-%.2f; phiSS %.2f-%.2f; phiS2S at T<=0.05 %.2f-%.2f; at 2 s %.2f' % (
    s.sigma_f.min(), s.sigma_f.max(), sm.sigma_f.min(), sm.sigma_f.max(), sm.tau_f.min(), sm.tau_f.max(), sm.phiSS_f.min(), sm.phiSS_f.max(),
    s.phiS2S_f[s['T'] <= 0.05 + 1e-9].min(), s.phiS2S_f[s['T'] <= 0.05 + 1e-9].max(), s.phiS2S_f['SA(2.000)_g']))
# conversion factor
c = cf[cf['T'].notna()]
i = c.CF.idxmin()
print('CF min %.3f at %s; periods with CF<=0.595: %s; 0.035-0.05 s %.3f-%.3f; 1.0 s %.3f; 2-3 s %.3f-%.3f; SD %.2f-%.2f (all IMs %.2f-%.2f)' % (
    c.CF.min(), i, list(c['T'][c.CF.round(2) <= 0.59].round(3)), c.CF[c['T'] <= 0.05 + 1e-9].min(), c.CF[c['T'] <= 0.05 + 1e-9].max(), c.CF['SA(1.000)_g'],
    c.CF[c['T'] >= 2 - 1e-9].min(), c.CF[c['T'] >= 2 - 1e-9].max(), c.sd_ln_station.min(), c.sd_ln_station.max(), cf.sd_ln_station.min(), cf.sd_ln_station.max()))
# station groups
print('BB 99-106 m: min %.3f at %s (%d stations); BB 60-67 m: min %.3f at %s (%d stations)' % (
    c.CF_bb100.min(), c.CF_bb100.idxmin(), c.n_st_bb100[c.CF_bb100.idxmin()], c.CF_bb65.min(), c.CF_bb65.idxmin(), c.n_st_bb65[c.CF_bb65.idxmin()]))
print('CF at 3 s %.3f' % c.CF['SA(3.000)_g'])
# characteristic periods
print('4h/VS: BB %.3f s, ACC %.3f s' % (4 * sb.mean() / site.reindex(sb.index).Vs_bb.mean(), 4 * st.h.mean() / s2.Vs_acc.mean()))
r = fin['PGA_g']['corrected']
sl = {}
for e, g in acc.groupby('event_id'):
    y = r.loc[g.index].values; x = g.Rjb_km.values
    if len(g) >= 3: sl[e] = np.polyfit(x, y, 1)[0] * 100
sl = pd.Series(sl)
pe = ev.index[ev.pohang_sequence]
print('PGA residual slope per 100 km: Pohang median %+.2f (all positive: %s); others median %+.2f' % (sl[pe].median(), bool((sl[pe] > 0).all()), sl.drop(pe, errors='ignore').median()))
