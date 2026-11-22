# EMOLO15, KHNP15, and JH22.
import numpy as np
import pandas as pd
from scipy.constants import g
from config import DATA


class EMOLO15():
    # Emolo et al. (2015)
    def __init__(self, imt, Repi_arr, ml_arr):
        self.coef = pd.read_csv(f'{DATA}/prev_models/EMOLO15.csv', index_col = 0)
        self.imt = imt
        self.ml_arr = ml_arr
        self.Repi_arr = Repi_arr
        if imt == 'PGV':
            self.unit_scale = 100
        else:
            self.unit_scale = 1/g

        supported_imt = ['PGV', 'PGA', 'SA(0.055)', 'SA(0.100)', 'SA(0.150)', 'SA(0.200)', 'SA(0.300)', 'SA(0.400)',
                         'SA(0.500)', 'SA(0.750)', 'SA(1.000)', 'SA(1.500)', 'SA(2.000)', 'SA(3.000)', 'SA(5.000)']
        if imt not in supported_imt:
            print(f'{supported_imt} supported')
            return
        self.a = self.coef.loc[imt, 'a'] * np.ones_like(Repi_arr)
        self.b = self.coef.loc[imt, 'b'] * np.ones_like(Repi_arr)
        self.c = self.coef.loc[imt, 'c'] * np.ones_like(Repi_arr)
        self.h = self.coef.loc[imt, 'h'] * np.ones_like(Repi_arr)
        self.d = self.coef.loc[imt, 'd'] * np.ones_like(Repi_arr)

    def _compute(self):
        logY = self.a + self.b * self.ml_arr + self.c * np.log10(np.sqrt(self.Repi_arr ** 2 + self.h ** 2)) + self.d * self.Repi_arr
        Y = 10 ** logY # unit is m/s, m/s^2
        Y *= self.unit_scale # convert from m/s, m/s^2 to cm/s, g
        return Y


class KHNP15():
    # KHNP (2015)
    def __init__(self, imt, Rhypo_arr, mw_arr, Repi = False, depth = None):
        self.coef = pd.read_csv(f'{DATA}/prev_models/KHNP15_03.csv', index_col = 0)
        self.imt = imt
        self.mw_arr = mw_arr
        self.Rhypo_arr = Rhypo_arr
        if Repi == True:
            self.Rhypo_arr = np.sqrt(self.Rhypo_arr ** 2 + depth ** 2)
        if imt == 'PGV':
            self.unit_scale = 1
        else:
            self.unit_scale = 1 / (g * 100)

        self.c0 = self.coef.loc[imt, 'xi00'] + self.coef.loc[imt, 'xi01'] * (mw_arr - 6) + self.coef.loc[imt, 'xi02'] * (mw_arr - 6) ** 2 + self.coef.loc[imt, 'xi03'] * (mw_arr - 6) ** 3
        self.c1 = self.coef.loc[imt, 'xi10'] + self.coef.loc[imt, 'xi11'] * (mw_arr - 6) + self.coef.loc[imt, 'xi12'] * (mw_arr - 6) ** 2 + self.coef.loc[imt, 'xi13'] * (mw_arr - 6) ** 3
        self.c2 = self.coef.loc[imt, 'xi20'] + self.coef.loc[imt, 'xi21'] * (mw_arr - 6) + self.coef.loc[imt, 'xi22'] * (mw_arr - 6) ** 2 + self.coef.loc[imt, 'xi23'] * (mw_arr - 6) ** 3

        self.c0 *= np.ones_like(Rhypo_arr)
        self.c1 *=  np.ones_like(Rhypo_arr)
        self.c2 *=  np.ones_like(Rhypo_arr)

    def _compute(self):
        min_max = 100 * np.ones_like(self.Rhypo_arr)
        min_max = np.c_[self.Rhypo_arr, min_max]
        lnY = self.c0 + self.c1 * self.Rhypo_arr + self.c2 * np.log(self.Rhypo_arr) - np.log(np.min(min_max, axis = 1)) - 1/2 * np.log(np.max(min_max, axis = 1))
        Y = np.exp(lnY)
        Y *= self.unit_scale
        return Y


class JH22():
    # Jee and Han (2022)
    # SA 0.501 assumed as SA 0.500
    # SA 1.995 assumed as SA 2.000
    # SA 5.012 assumed as SA 5.000

    def __init__(self, imt, Rhypo_arr, mw_arr, Repi = False, depth = None):
        self.coef = pd.read_csv(f'{DATA}/prev_models/JH22.csv', index_col = 0)
        self.imt = imt
        self.mw_arr = mw_arr
        self.Rhypo_arr = Rhypo_arr
        if Repi == True:
            self.Rhypo_arr = np.sqrt(self.Rhypo_arr ** 2 + depth ** 2)

        self.c1 = np.ones_like(self.Rhypo_arr) * self.coef.loc[imt, 'c1']
        self.c2 = np.ones_like(self.Rhypo_arr) * self.coef.loc[imt, 'c2']
        self.c3 = np.ones_like(self.Rhypo_arr) * self.coef.loc[imt, 'c3']
        self.c4 = np.ones_like(self.Rhypo_arr) * self.coef.loc[imt, 'c4']
        self.c5 = np.ones_like(self.Rhypo_arr) * self.coef.loc[imt, 'c5']
        self.c6 = np.ones_like(self.Rhypo_arr) * self.coef.loc[imt, 'c6']
        self.c7 = np.ones_like(self.Rhypo_arr) * self.coef.loc[imt, 'c7']
        self.c8 = np.ones_like(self.Rhypo_arr) * self.coef.loc[imt, 'c8']
        self.c9 = np.ones_like(self.Rhypo_arr) * self.coef.loc[imt, 'c9']
        self.c10 = np.ones_like(self.Rhypo_arr) * self.coef.loc[imt, 'c10']

    def _compute(self):
        logY = (
            self.c1 + self.c2 * self.mw_arr + self.c3 * self.mw_arr**2 + (self.c4 + self.c5 * self.mw_arr) * np.minimum(np.log10(self.Rhypo_arr), np.log10(50))
            + (self.c6 + self.c7 * self.mw_arr) * np.maximum(np.minimum(np.log10(self.Rhypo_arr / 50), np.log10(100 / 50)), 1e-12)
            + (self.c8 + self.c9 * self.mw_arr) * np.maximum(np.log10(self.Rhypo_arr / 100), 1e-12) + self.c10 * self.Rhypo_arr
            )

        Y = 10 ** logY # unit is g
        return Y
