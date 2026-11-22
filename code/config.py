# Paths and constants.
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'results')     # tables
WORK = os.path.join(ROOT, 'work')       # intermediate files
FIG = os.path.join(ROOT, 'figures')
for d in (OUT, WORK, FIG):
    os.makedirs(d, exist_ok=True)

FAC = 1.25       # usable periods: 1.25/f_lp <= T <= 1/(1.25 f_hp)
TOL = 1e-12
MCAP = 5.7       # magnitude cap of equation (3)
MIN_REC = 3      # minimum usable records per event at a period
GY = 'kr2016256113254'    # 2016 Gyeongju mainshock
MW_SWITCH = '2019-07-01'  # new ML formula of the KMA


def mw_from_ml(ml, origin_time_utc):
    # Sheen et al. (2018): equation 10 before the switch, then equation 8 (vertical)
    m = float(ml)
    if pd.to_datetime(origin_time_utc, utc=True) < pd.Timestamp(MW_SWITCH, tz='UTC'):
        m = 0.9234 * m + 0.3262
    return (m + 0.4772) / 1.086
