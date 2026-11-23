# Data and code

K. Jo, H. Yoo, Y. Jeon, D. Park, S. Choi, and S. Kim, "Borehole-Based Referenced Empirical Ground-Motion Model for the
Southern Korean Peninsula and the Depth Dependence of Within-Rock Motions".
Abbreviations are those of the article.

## Folders
- data/: input data.
- code/: scripts.
- results/: output tables.
- figures/: Figures 1-18 and S1-S5, and captions.txt.

## Data
- Flatfile_ACC.xlsx: 9,103 accelerometer records of 114 events at 275 stations.
- Flatfile_BB.xlsx: 3,201 broadband records at 93 stations.
- Flatfile columns: PGA and SA in g, PGV in cm/s, RotD50. SA is 5%-damped, at 104 periods from 0.02 to 10 s.
- Site_metadata.xlsx: sensor depths, bedrock depths, VS30, and VS at the sensor depths.
- vs_profiles/: VS-profile ensembles of Jeon et al. (2026) at GKP2, CHRB, CIGB, and EURB. 3,000 profiles each.
  A profile is the number of layers followed by one line per layer. Columns 1 and 3 are thickness (m) and VS (m/s).
- waveforms/: accelerometer records (MiniSEED, E and N) and StationXML files of the four stations. file_list.csv lists
  them.
- prev_models/: coefficients of EMOLO15, KHNP15, and JH22.

## Run
- cd code, then sh run_all.sh. About 45 minutes.
- Intermediate files are written to work/.
- numbers_*.py and tables_1_2.py print the values given in the article.

## Scripts
- s1_dataset.py: MW and residuals relative to BSSA14.
- s2_fit.py: regressions of equation (3).
- s2_post.py: smoothing, standard deviations, conversion factors.
- screening_stages.py: conversion factor at each screening stage.
- s3_withheld.py, s3_bootstrap.py: withholding tests.
- s4_additional.py: station-level predictors and leave-one-event-out fits.
- s5_onedim_tf.py, s5_onedim_syn.py: 1D transfer functions and synthetic records.
- s6_stoch.py: stochastic simulations.
- s7_magnitude_terms.py: quadratic and bilinear magnitude terms.
- s8_tables.py: Tables S2-S4 and S7.
- fig_*.py: figures.

## Results
- events.csv: events and MW.
- coeff_main.csv: period-by-period fit of the adopted dataset.
- coeff_main_smoothed.csv: adopted model. a_s, a1_s, and b1_s are the coefficients. tau_f to sigma_f are the standard
  deviations.
- coeff_m4.csv, coeff_pohang_in.csv, coeff_gy55.csv: MW >= 4 only, Pohang sequence included, Gyeongju MW 5.5.
- conv_factors.csv, bb_variability.csv, screening_stages.csv: conversion factors and BB-level statistics.
- withheld_rmse.csv, withheld_mw4.csv, withheld_bootstrap.csv: Tables S5 and S6.
- station_predictor_tests.csv, jackknife_b1_m4.csv, magnitude_terms.csv: additional regressions.
- onedim_tf_summary*.csv, onedim_syn_*.csv: 1D analysis. Q100 and Q10 are QS = VS/100 and VS/10.
- stochastic_PGM_WNA.csv, stochastic_PGM_SKP.csv: stochastic predictions for each record.
- GMM_supplement_tables.xlsx: Tables S2-S4 and S7.
- Residuals, terms, and standard deviations are in natural-log units.

## Software
Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0, matplotlib 3.11, basemap 2.0, obspy 1.5, pygmm 0.8.0, pyrvt 0.8.1,
openpyxl 3.1, xlsxwriter 3.2. The figures use the Liberation Sans font.

## Sources
- Records and station metadata: Korea Meteorological Administration and Korea Institute of Geoscience and Mineral
  Resources.
- Jeon, Y., S. Kim, L. Joshi, and H.-S. Kim (2026). Bull. Earthq. Eng. 24, 1853-1881.
