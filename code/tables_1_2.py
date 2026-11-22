# Tables 1 and 2 of the article.
import pandas as pd
from config import WORK

IMS = ['PGV_cm_s', 'PGA_g'] + ['SA(%.3f)_g' % t for t in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0)]
sm = pd.read_pickle(f'{WORK}/coeff_main_smoothed.pkl').set_index('IM').loc[IMS]
cf = pd.read_pickle(f'{WORK}/conv_factors.pkl').set_index('IM').loc[IMS]
bb = pd.read_pickle(f'{WORK}/bb_variability.pkl').set_index('IM').loc[IMS]
pd.set_option('display.width', 200)
t1 = sm[['a_s', 'a1_s', 'b1_s', 'tau_f', 'phiS2S_f', 'phiSS_f', 'phi_f', 'sigma_f', 'n', 'n_events']]
print('Table 1'); print(t1.round({'a_s': 3, 'a1_s': 4, 'b1_s': 3, 'tau_f': 3, 'phiS2S_f': 3, 'phiSS_f': 3, 'phi_f': 3, 'sigma_f': 3}).to_string())
t2 = cf[['CF', 'sd_ln_station', 'n_stations', 'n_pairs']].join(bb[['tau', 'phiS2S', 'phiSS', 'phi', 'sigma']])
print('Table 2'); print(t2.round(3).to_string())
