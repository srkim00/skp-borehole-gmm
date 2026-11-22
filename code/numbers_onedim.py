# Numbers of the one-dimensional analysis quoted in the text.
import sys
import numpy as np, pandas as pd
from config import OUT, WORK

STN = ['GKP2', 'CHRB', 'CIGB', 'EURB']
q = sys.argv[1] if len(sys.argv) > 1 else 'Q100'
tfz = np.load(f'{WORK}/onedim_tf.npz' if q == 'Q100' else f'{WORK}/onedim_tf_{q}.npz'); F = tfz['freq']; T = 1 / F
stm = pd.read_pickle(f'{WORK}/station_mean_lnratio.pkl')
syn = pd.read_csv(f'{OUT}/onedim_syn_station_means_{q}.csv')
cols = [c for c in stm.columns if c.startswith('SA(')]; Tobs = np.array([float(c[3:-3]) for c in cols])
print('station  TFmedian_min (period)  obs_min (period)  syn_min (period)  TF@1s  obs@1s  syn@1s  minper p16-p84')
for s in STN:
    rms = tfz[f'{s}_{q}_rms']; med = np.median(rms, axis=0)
    k = (T >= 0.03) & (T <= 1.0); i = np.argmin(med[k]); tf_min, tf_T = med[k][i], T[k][i]
    pm = T[k][np.argmin(rms[:, k], axis=1)]
    o = np.exp(stm.loc[s, cols].astype(float).values); ko = (Tobs >= 0.03) & (Tobs <= 1.0) & np.isfinite(o)
    j = np.nanargmin(np.where(ko, o, np.nan))
    d = syn[syn.station == s]; ks = (d['T'] >= 0.03) & (d['T'] <= 1.0); js = d.ratio[ks].idxmin()
    o1 = o[np.argmin(abs(Tobs - 1.0))]; s1 = d.ratio.values[np.argmin(abs(d['T'].values - 1.0))]
    print('%-6s %.3f (%.3f s)  %.3f (%.3f s)  %.3f (%.3f s)  %.3f  %.3f  %.3f  %.3f-%.3f (width %.3f)' % (
        s, tf_min, tf_T, o[j], Tobs[j], d.ratio[js], d['T'][js], med[np.argmin(abs(T - 1.0))], o1, s1,
        np.percentile(pm, 16), np.percentile(pm, 84), np.percentile(pm, 84) - np.percentile(pm, 16)))
