# Synthetic records at the broadband depths and their SA ratios.
import os, sys, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import obspy
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.rotate import rotate_ne_rt
from config import DATA, OUT, WORK, FAC, TOL

STN = ['GKP2', 'CHRB', 'CIGB', 'EURB']
WF = os.path.join(DATA, 'waveforms')
PRE_FILT = (0.05, 0.1, 40.0, 45.0); WATER = 60
ANG = np.deg2rad(np.arange(0, 180, 1))
XI = 0.05


def rotd50(a, b, dt, periods):
    npts = len(a); nfft = int(2 ** np.ceil(np.log2(npts * 2)))
    f = np.fft.rfftfreq(nfft, dt); w = 2 * np.pi * f
    A = np.fft.rfft(a, nfft); B = np.fft.rfft(b, nfft)
    out = np.empty(len(periods)); c = np.cos(ANG)[:, None]; s = np.sin(ANG)[:, None]
    for i, T in enumerate(periods):
        w0 = 2 * np.pi / T
        H = -w0 ** 2 / (w0 ** 2 - w ** 2 + 2j * XI * w0 * w)   # numpy FFT convention exp(-i w t)
        ra = np.fft.irfft(A * H, nfft)[:npts]; rb = np.fft.irfft(B * H, nfft)[:npts]
        out[i] = np.median(np.max(np.abs(c * ra + s * rb), axis=1))
    return out


def apply_amp(x, dt, tf, tf_f):
    npts = len(x); nfft = int(2 ** np.ceil(np.log2(npts * 2)))
    f = np.fft.rfftfreq(nfft, dt)
    g = np.interp(np.log(np.maximum(f, 1e-6)), np.log(tf_f), tf, left=tf[0], right=tf[-1])
    return np.fft.irfft(np.fft.rfft(x, nfft) * g, nfft)[:npts]


def process(fe, fn, inv, fhp):
    st = obspy.Stream([obspy.read(fe)[0], obspy.read(fn)[0]])
    st.detrend('demean'); st.detrend('linear'); st.taper(0.05)
    st.remove_response(inventory=inv, output='ACC', pre_filt=PRE_FILT, water_level=WATER)
    st.filter('highpass', freq=fhp, corners=4, zerophase=True)
    npts = min(len(st[0].data), len(st[1].data))
    return st[0].data[:npts].astype(float), st[1].data[:npts].astype(float), st[0].stats.delta


if __name__ == '__main__':
    qtag = sys.argv[1] if len(sys.argv) > 1 else 'Q100'
    acc = pd.read_pickle(f'{WORK}/acc.pkl')
    sa_cols = [c for c in acc.columns if c.startswith('SA(')]
    periods = np.array([float(c[3:-3]) for c in sa_cols])
    tfz = np.load(f'{WORK}/onedim_tf.npz' if qtag == 'Q100' else f'{WORK}/onedim_tf_{qtag}.npz'); FREQ = tfz['freq']
    files = pd.read_csv(os.path.join(WF, 'file_list.csv'))
    rows = []
    for stn in STN:
        inv = obspy.read_inventory(glob.glob(os.path.join(WF, 'stationxml', f'*.{stn}.xml'))[0])
        tf_sv = np.median(tfz[f'{stn}_{qtag}_sv'], axis=0); tf_sh = np.median(tfz[f'{stn}_{qtag}_sh'], axis=0)
        for _, r in files[files.station_code == stn].iterrows():
            rec = acc[(acc.event_id == r.event_id) & (acc.station_code == stn)].iloc[0]
            e, n, dt = process(os.path.join(WF, stn, r.file_E), os.path.join(WF, stn, r.file_N), inv, rec.highpass_hz)
            baz = gps2dist_azimuth(rec.station_latitude_deg, rec.station_longitude_deg, rec.event_latitude_deg, rec.event_longitude_deg)[2]
            R, Tt = rotate_ne_rt(n, e, baz)
            Rs = apply_amp(R, dt, tf_sv, FREQ); Ts = apply_amp(Tt, dt, tf_sh, FREQ)
            sa_a = rotd50(R, Tt, dt, periods); sa_s = rotd50(Rs, Ts, dt, periods)
            row = dict(station=stn, event_id=r.event_id, highpass_hz=rec.highpass_hz, lowpass_hz=rec.lowpass_hz,
                       T_min=FAC / rec.lowpass_hz, T_max=1 / (FAC * rec.highpass_hz))
            row.update({'acc_%g' % T: v for T, v in zip(periods, sa_a)}); row.update({'syn_%g' % T: v for T, v in zip(periods, sa_s)})
            rows.append(row)
        print(stn, 'records', sum(1 for x in rows if x['station'] == stn), flush=True)
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/onedim_syn_records_{qtag}.csv', index=False)
    means = []
    for stn in STN:
        d = df[df.station == stn]
        for T in periods:
            ok = (T <= d.T_max + TOL) & (T >= d.T_min - TOL)
            if ok.sum() >= 3:
                lr = np.log(d['syn_%g' % T][ok] / d['acc_%g' % T][ok])
                means.append(dict(station=stn, T=T, n=int(ok.sum()), ratio=float(np.exp(lr.mean()))))
    pd.DataFrame(means).to_csv(f'{OUT}/onedim_syn_station_means_{qtag}.csv', index=False)
    print('done')
