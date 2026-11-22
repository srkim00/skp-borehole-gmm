# Transfer functions between the two sensor depths at four stations.
import os, sys, time
import numpy as np, pandas as pd
from onedim_tf import read_models, tf_oblique, ko_smooth
from config import DATA, OUT, WORK

STN = ['GKP2', 'CHRB', 'CIGB', 'EURB']
DEPTH = {'GKP2': (30.0, 100.0), 'CHRB': (26.0, 100.0), 'CIGB': (23.0, 100.0), 'EURB': (20.0, 100.0)}
FREQ = np.geomspace(0.05, 50.0, 700)
SEED = 20261115
NPROF = 100

if __name__ == '__main__':
    qfacs = [float(x) for x in sys.argv[1:]] or [100.0]
    rng = np.random.default_rng(SEED)
    picks = {s: np.sort(rng.choice(3000, NPROF, replace=False)) for s in STN}
    out = {'freq': FREQ}
    rows = []
    t0 = time.time()
    for q in qfacs:
        for stn in STN:
            models = read_models(os.path.join(DATA, 'vs_profiles', f'{stn}_model.txt'))
            za, zb = DEPTH[stn]
            sv = np.zeros((NPROF, len(FREQ))); sh = np.zeros((NPROF, len(FREQ)))
            for i, k in enumerate(picks[stn]):
                a, b = tf_oblique(models[k], za, zb, FREQ, qfac=q)
                sv[i] = ko_smooth(a, FREQ); sh[i] = ko_smooth(b, FREQ)
            rms = np.sqrt((sv ** 2 + sh ** 2) / 2)
            tag = f'{stn}_Q{int(q)}'
            out[tag + '_sv'] = sv; out[tag + '_sh'] = sh; out[tag + '_rms'] = rms; out[tag + '_idx'] = picks[stn]
            T = 1 / FREQ; k = (T >= 0.03) & (T <= 1.0)
            med = np.median(rms, axis=0)
            per_min = T[k][np.argmin(rms[:, k], axis=1)]
            rows.append(dict(station=stn, QS='VS/%d' % q, median_min=med[k].min(), period_of_median_min=T[k][np.argmin(med[k])],
                             median_at_1s=med[np.argmin(abs(T - 1.0))], pmin_p16=np.percentile(per_min, 16), pmin_p84=np.percentile(per_min, 84)))
            print(tag, 'minimum %.3f at %.3f s' % (rows[-1]['median_min'], rows[-1]['period_of_median_min']), round(time.time() - t0), 's', flush=True)
    tag = '' if qfacs == [100.0] else '_' + '_'.join('Q%d' % q for q in qfacs)
    np.savez_compressed(f'{WORK}/onedim_tf{tag}.npz', **out)
    pd.DataFrame(rows).to_csv(f'{OUT}/onedim_tf_summary{tag}.csv', index=False)
    print('done', round(time.time() - t0), 's')
