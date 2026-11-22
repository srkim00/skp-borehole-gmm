# Stochastic simulations with the WNA and SKP attenuation (Table 3).
import sys, time, pickle
import numpy as np, pandas as pd
from scipy.interpolate import interp1d
import pyrvt.motions as pm, pyrvt.peak_calculators as pc
from multiprocessing import Pool
from config import OUT, WORK

GRAV = 980.665
PARAMS = {'WNA': dict(q0=202.0, eta=0.54), 'SKP': dict(q0=229.2, eta=0.73)}
IMS = [('PGV_cm_s', None), ('PGA_g', None), ('SA(0.050)_g', 0.05), ('SA(0.100)_g', 0.1), ('SA(0.300)_g', 0.3),
       ('SA(1.000)_g', 1.0), ('SA(2.000)_g', 2.0), ('SA(3.000)_g', 3.0)]
FREQS = np.geomspace(0.02, 100.0, 1200)


def motion(mag, r, q0, eta, stress=135.0):
    mo = pm.SourceTheoryMotion(mag, r, 'wna', stress_drop=stress, depth=0.0,
                               peak_calculator=pc.BooreThompson2015('wna', mag, max(r, 1.0)), freqs=FREQS)
    mo.shear_velocity = 3.5; mo.density = 2.8
    mo.path_atten_coeff = q0; mo.path_atten_power = eta
    mo.site_atten = 0.035
    mo.geometric_spreading = [(1.03, 45.0), (0.96, 125.0), (0.5, None)]
    mo.site_amp = interp1d(np.log([0.001, 1000.0]), [1.0, 1.0], bounds_error=False, fill_value=(1.0, 1.0))
    mo.hypo_distance = max(r, 1.0)
    mo.calc_fourier_amps(FREQS)
    return mo


def simulate(args):
    mag, rhyp, key = args
    p = PARAMS[key]
    mo = motion(mag, rhyp, p['q0'], p['eta'])
    out = {'PGA_g': float(mo.calc_peak()), 'PGV_cm_s': float(mo.calc_peak(transfer_func=1.0 / (2 * np.pi * mo.freqs))) * GRAV}
    sa = mo.calc_osc_accels([1.0 / t for _, t in IMS if t is not None], 0.05)
    for (im, t), v in zip([x for x in IMS if x[1] is not None], sa):
        out[im] = float(v)
    return out


if __name__ == '__main__':
    acc = pd.read_pickle(f'{WORK}/acc.pkl')
    t0 = time.time(); res = {}
    for key in ['WNA', 'SKP']:
        args = [(float(m), float(r), key) for m, r in zip(acc.Mw.values, acc.Rhypo_km.values)]
        with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
            rows = pool.map(simulate, args, chunksize=50)
        res[key] = pd.DataFrame(rows, index=acc.index)
        o = acc[['event_id', 'station_code', 'Mw', 'Rhypo_km']].rename(columns={'Mw': 'MW'}).join(res[key])
        o.to_csv(f'{OUT}/stochastic_PGM_{key}.csv', index=False)
        print(key, 'done', round(time.time() - t0, 1), 's', flush=True)
    with open(f'{WORK}/stoch_psa.pkl', 'wb') as fh:
        pickle.dump(res, fh)
