# Supplement workbook (Tables S2, S3, S4, S7).
import os, json, pickle
import numpy as np, pandas as pd
from config import OUT, WORK, DATA, FAC
from cf_windows import EXCL

DEST = os.path.join(OUT, 'GMM_supplement_tables.xlsx')
acc = pd.read_pickle(f'{WORK}/acc.pkl'); bb = pd.read_pickle(f'{WORK}/bb.pkl')
site = pd.read_excel(os.path.join(DATA, 'Site_metadata.xlsx')).set_index('Station')
p1 = pd.read_pickle(f'{WORK}/pairs_kept.pkl')
sm = pd.read_pickle(f'{WORK}/coeff_main_smoothed.pkl'); cf = pd.read_pickle(f'{WORK}/conv_factors.pkl'); bbv = pd.read_pickle(f'{WORK}/bb_variability.pkl')
stm = pd.read_pickle(f'{WORK}/station_mean_lnratio.pkl'); sts = pd.read_pickle(f'{WORK}/station_sd_lnratio.pkl'); stn = pd.read_pickle(f'{WORK}/station_n_pairs.pkl')
with open(f'{WORK}/final_effects.pkl', 'rb') as fh:
    final = pickle.load(fh)
with open(f'{WORK}/screen_detail.pkl', 'rb') as fh:
    detail = pickle.load(fh)
IMS = sm.IM.tolist(); TIM = sm['T'].values
IMname = {'PGV_cm_s': 'PGV', 'PGA_g': 'PGA'}
for im in IMS:
    IMname.setdefault(im, im.replace('_g', ''))
COORD = ('latitude', 'longitude', 'event_latitude_deg', 'event_longitude_deg')


def sig(df, n=5):
    f = np.vectorize(lambda v: float('%.*g' % (n, v)) if np.isfinite(v) else np.nan, otypes=[float])
    for c in df.columns:
        if c in COORD:
            df[c] = df[c].round(4)
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = f(df[c].values)
    return df


ev = acc.groupby('event_id').agg(lat=('event_latitude_deg', 'first'), lon=('event_longitude_deg', 'first'))
pohang = set(ev[(ev.lat.between(35.9, 36.3)) & (ev.lon.between(129.2, 129.5))].index)

kept_keys = set(zip(p1.event_id, p1.station_code))
removed_at = {k: [] for k in kept_keys}
for im in IMS:
    for tag, key in (('ratio range', 'range'), ('station', 'station'), ('record', 'record')):
        for i in detail[im][key]:
            removed_at[(p1.at[i, 'event_id'], p1.at[i, 'station_code'])].append('%s (%s)' % (IMname[im], tag))

sa_cols = [c for c in acc.columns if c.startswith('SA(')]
R = pd.DataFrame({
    'event_id': acc.event_id.values, 'origin_time_utc': acc.origin_time_utc.values,
    'event_latitude_deg': acc.event_latitude_deg.values, 'event_longitude_deg': acc.event_longitude_deg.values,
    'event_depth_km': acc.event_depth_km.values, 'ML': acc.ML.values, 'MW': acc.Mw.values,
    'pohang_sequence': acc.event_id.isin(pohang).values, 'regression_dataset': ~acc.event_id.isin(pohang).values,
    'station_code': acc.station_code.values, 'network': acc.network.values,
    'Rjb_km': acc.Rjb_km.values, 'Repi_km': acc.Repi_km.values, 'Rhypo_km': acc.Rhypo_km.values,
    'acc_sensor_depth_m': acc.sensor_depth_m.values, 'acc_highpass_hz': acc.highpass_hz.values, 'acc_lowpass_hz': acc.lowpass_hz.values,
    'acc_T_min_s': FAC / acc.lowpass_hz.values, 'acc_T_max_s': 1.0 / (FAC * acc.highpass_hz.values),
    'acc_PGA_g': acc.PGA_g.values, 'acc_PGV_cm_s': acc.PGV_cm_s.values})
extra = {'acc_' + c: acc[c].values for c in sa_cols}
bbi = bb.set_index(['event_id', 'station_code']); keys = list(zip(acc.event_id, acc.station_code))
has_bb = np.array([k in bbi.index for k in keys]); bbr = bbi.reindex(keys)
extra.update({'bb_sensor_depth_m': bbr.sensor_depth_m.values, 'bb_highpass_hz': bbr.highpass_hz.values, 'bb_lowpass_hz': bbr.lowpass_hz.values,
              'bb_T_min_s': FAC / bbr.lowpass_hz.values, 'bb_T_max_s': 1.0 / (FAC * bbr.highpass_hz.values),
              'bb_PGA_g': bbr.PGA_g.values, 'bb_PGV_cm_s': bbr.PGV_cm_s.values})
extra.update({'bb_' + c: bbr[c].values for c in sa_cols})
status = np.where(~has_bb, 'no broadband record', np.where([k in kept_keys for k in keys], 'retained', 'excluded (time interval)'))
extra['pair_status'] = status
extra['pair_removed_at'] = ['; '.join(removed_at[k]) if k in removed_at else '' for k in keys]
for im in IMS:
    extra['event_term_' + IMname[im]] = acc.event_id.map(final[im]['eta']).values
R = sig(pd.concat([R, pd.DataFrame(extra)], axis=1))
R.loc[R.pohang_sequence, [c for c in R.columns if c.startswith('event_term_')]] = np.nan

st = acc.groupby('station_code').agg(network=('network', 'first'), latitude=('station_latitude_deg', 'first'),
                                     longitude=('station_longitude_deg', 'first'), elevation_m=('station_elevation_m', 'first'),
                                     acc_depth_ff=('sensor_depth_m', 'first'), n_acc_records=('event_id', 'size'))
st = st.join(bb.groupby('station_code').agg(n_bb_records=('event_id', 'size')))
s = site.reindex(st.index)
pair_all = acc.merge(bb, on=['event_id', 'station_code'])
n_all = pair_all.groupby('station_code').size(); n_kept = p1.groupby('station_code').size()
hasbb = st.n_bb_records.fillna(0) > 0
cols = {'network': st.network, 'latitude': st.latitude, 'longitude': st.longitude, 'elevation_m': st.elevation_m,
        'acc_sensor': s.Sensor_ACC, 'acc_sensor_depth_m': s.Depth_ACC.where(s.Depth_ACC.notna(), st.acc_depth_ff),
        'bb_sensor': s.Sensor_BB, 'bb_sensor_depth_m': s.Depth_BB, 'recorder': s.Recorder,
        'n_acc_records': st.n_acc_records, 'n_bb_records': st.n_bb_records.fillna(0).astype(int),
        'bedrock_depth_m': s.H_geo, 'VS30_m_s': s.Vs30, 'VS_acc_m_s': s.Vs_acc, 'VS_bb_m_s': s.Vs_bb,
        'pairs_excluded_period_utc': pd.Series(EXCL).reindex(st.index),
        'pairs_excluded': (n_all.reindex(st.index).fillna(0) - n_kept.reindex(st.index).fillna(0)).where(hasbb),
        'pairs_retained': n_kept.reindex(st.index).fillna(0).where(hasbb)}
for im in IMS:
    cols['station_term_' + IMname[im]] = final[im]['dS'].reindex(st.index)
for im in IMS:
    cols['ratio_' + IMname[im]] = np.exp(stm[im].reindex(st.index))
for im in IMS:
    cols['ratio_sd_ln_' + IMname[im]] = sts[im].reindex(st.index)
for im in IMS:
    cols['ratio_pairs_' + IMname[im]] = stn[im].reindex(st.index)
S2 = pd.DataFrame(cols, index=st.index); S2.index.name = 'station_code'
S2 = sig(S2.reset_index())

S4 = pd.DataFrame({
    'IM': [IMname[i] for i in IMS], 'Period_s': TIM,
    'a (adopted, smoothed)': sm.a_s.values, 'a1 (adopted, smoothed)': sm.a1_s.values, 'b1 (adopted, smoothed)': sm.b1_s.values,
    'tau (adopted)': sm.tau_f.values, 'phi_S2S (adopted)': sm.phiS2S_f.values, 'phi_SS (adopted)': sm.phiSS_f.values,
    'phi (adopted)': sm.phi_f.values, 'sigma (adopted)': sm.sigma_f.values,
    'a (period-by-period)': sm.a.values, 'a1 (period-by-period)': sm.a1.values, 'b1 (period-by-period)': sm.b1.values,
    'SE a': sm.se_a.values, 'SE a1': sm.se_a1.values, 'SE b1': sm.se_b1.values,
    'tau (period-by-period)': sm.tau.values, 'phi_S2S (period-by-period)': sm.phiS2S.values, 'phi_SS (period-by-period)': sm.phiSS.values,
    'phi (period-by-period)': sm.phi.values, 'sigma (period-by-period)': sm.sigma.values,
    'Records (ACC level)': sm.n.values, 'Events (ACC level)': sm.n_events.values, 'Stations (ACC level)': sm.n_stations.values,
    'MW min (ACC level)': sm.Mw_min.values, 'MW max (ACC level)': sm.Mw_max.values,
    'Conversion factor (geometric mean of station means)': cf.CF.values, 'SD of station ln-means': cf.sd_ln_station.values,
    'Stations (conversion factor)': cf.n_stations.values, 'Pairs (conversion factor)': cf.n_pairs.values,
    'Pairs removed (ratio outside 0.2-5)': cf.removed_range.values, 'Stations removed (station-level screening)': cf.removed_stations.values,
    'Records removed (record-level screening)': cf.removed_records.values,
    'Factor, broadband sensor at 99-106 m': cf.CF_bb100.values, 'Stations (99-106 m)': cf.n_st_bb100.values,
    'Factor, broadband sensor at 60-67 m': cf.CF_bb65.values, 'Stations (60-67 m)': cf.n_st_bb65.values,
    'BB-level mean residual (ln)': bbv.bias.values, 'SE of mean residual': bbv.se_bias.values,
    'tau (BB level)': bbv.tau.values, 'phi_S2S (BB level)': bbv.phiS2S.values, 'phi_SS (BB level)': bbv.phiSS.values,
    'phi (BB level)': bbv.phi.values, 'sigma (BB level)': bbv.sigma.values,
    'Records (BB level)': bbv.n.values, 'Events (BB level)': bbv.n_events.values, 'Stations (BB level)': bbv.n_stations.values})
assert (cf.IM.values == sm.IM.values).all() and (bbv.IM.values == sm.IM.values).all()
S4 = sig(S4)

SENS = [('main', 'All events (Pohang sequence excluded; adopted dataset)'), ('m4', 'Events with MW >= 4.0 only'),
        ('pohang_in', 'All events, Pohang sequence included'), ('gy55', 'All events, Gyeongju mainshock MW = 5.5')]
rows = []
for k, labl in SENS:
    for _, r in pd.read_pickle(f'{WORK}/coeff_{k}.pkl').iterrows():
        rows.append({'Test': 'Sensitivity to the dataset', 'IM': IMname[r.IM], 'Period_s': r['T'], 'Dataset or model': labl, 'Records': r.n,
                     'Events': r.n_events, 'a': r.a, 'a1': r.a1, 'b1': r.b1, 'SE_a': r.se_a, 'SE_a1': r.se_a1, 'SE_b1': r.se_b1, 'tau': r.tau,
                     'phi_S2S': r.phiS2S, 'phi_SS': r.phiSS, 'phi': r.phi, 'sigma': r.sigma})
lab = {'base (equation 3)': 'Base model (89 stations with VS profiles)', '+ ln(h/23 m)': 'Base model + ln(sensor depth/23 m)',
       '+ ln(VS30/600 m/s)': 'Base model + ln(VS30/600 m/s)', '+ ln(VS_acc/1900 m/s)': 'Base model + ln(VS at the sensor depth/1,900 m/s)'}
for _, r in pd.read_csv(f'{OUT}/station_predictor_tests.csv').iterrows():
    rows.append({'Test': 'Station-level predictors as fixed effects', 'IM': IMname[r.IM], 'Period_s': r['T'], 'Dataset or model': lab[r.model],
                 'Records': r.n, 'Events': r.n_events, 'a1': r.a1, 'b1': r.b1, 'tau': r.tau, 'phi_S2S': r.phiS2S, 'phi_SS': r.phiSS, 'sigma': r.sigma,
                 'c (added term)': r.get('c'), 'SE c': r.get('se_c'), 'p (likelihood ratio)': r.get('p'), 'Change of phi_S2S (%)': r.get('dphiS2S_pct')})
cols7 = ['Test', 'IM', 'Period_s', 'Dataset or model', 'Records', 'Events', 'a', 'a1', 'b1', 'SE_a', 'SE_a1', 'SE_b1', 'tau',
         'phi_S2S', 'phi_SS', 'phi', 'sigma', 'c (added term)', 'SE c', 'p (likelihood ratio)', 'Change of phi_S2S (%)']
S7 = sig(pd.DataFrame(rows).reindex(columns=cols7).astype({c: float for c in cols7[4:]}))

README = pd.DataFrame(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables_readme.json'))),
                      columns=['Table', 'Column(s)', 'Description'])

with pd.ExcelWriter(DEST, engine='xlsxwriter') as w:
    README.to_excel(w, sheet_name='README', index=False)
    S2.to_excel(w, sheet_name='TableS2_stations', index=False)
    R.to_excel(w, sheet_name='TableS3_records', index=False)
    S4.to_excel(w, sheet_name='TableS4_model', index=False)
    S7.to_excel(w, sheet_name='TableS7_additional_regressions', index=False)
    wrap = w.book.add_format({'text_wrap': True, 'valign': 'top'})
    ws = w.sheets['README']; ws.set_column(0, 0, 10, wrap); ws.set_column(1, 1, 45, wrap); ws.set_column(2, 2, 110, wrap)
    for name in ('TableS2_stations', 'TableS3_records', 'TableS4_model', 'TableS7_additional_regressions'):
        w.sheets[name].freeze_panes(1, 1)
print('written', DEST, os.path.getsize(DEST), 'bytes')
print('README', README.shape, 'S2', S2.shape, 'S3', R.shape, 'S4', S4.shape, 'S7', S7.shape)
print('pair status:', pd.Series(status).value_counts().to_dict())
