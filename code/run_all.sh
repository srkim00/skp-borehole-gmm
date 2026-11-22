#!/bin/sh
# Analysis, tables, and figures (about 45 minutes).
set -e
python3 s1_dataset.py
python3 s2_fit.py
python3 s2_post.py
python3 screening_stages.py
python3 s3_withheld.py
python3 s3_bootstrap.py
python3 s4_additional.py
python3 s5_onedim_tf.py
python3 s5_onedim_tf.py 10
python3 s5_onedim_syn.py Q100
python3 s5_onedim_syn.py Q10
python3 s6_stoch.py
python3 s7_magnitude_terms.py
python3 s8_tables.py
for f in fig_*.py; do python3 $f; done
