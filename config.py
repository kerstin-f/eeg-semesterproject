"""
===========
Config file
===========

Configuration parameters for the study.
"""

import os
import getpass
from socket import getfqdn
from typing import Optional

from numpy import NaN
from fnames import FileNames


###############################################################################
# Determine which user is running the scripts on which machine and set the path
# where the data is stored and how many CPU cores to use.

user = getpass.getuser()  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts

# You want to add your machine to this list
if user == 'kerst':
    # My laptop
    raw_data_dir = './data'
    n_jobs = 1
else:
    # Defaults
    raw_data_dir = './data'
    n_jobs = 1

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)


###############################################################################
# These are all the relevant parameters for the analysis.

bids_root = "./data/p3/"
task = "p3" 

# # All subjects from 1 to N
# count = 0
# for _, dirs, _ in os.walk(bids_root):
# 	for dir in dirs:
# 		if(dir.startswith('sub-')):
# 			count = count + 1
# subjects = ["%.3d" % i for i in range(1,count+1)]

subjects = ['001', '002', '003']


###############################################################################
# Frequency filtering
h_freq = 0.5
l_freq = 50
h_trans_bandwidth = None
l_trans_bandwidth = None

###############################################################################
# Epoching
conditions = ['stimulus']
epochs_metadata_tmin = -0.1
"""
The beginning of the time window for metadata generation, in seconds,
relative to the time-locked event of the respective epoch. This may be less
than or larger than the epoch's first time point. If ``None``, use the first
time point of the epoch.
"""
epochs_metadata_tmax = 1
"""
Same as ``epochs_metadata_tmin``, but specifying the **end** of the time
window for metadata generation.
"""

eeg_reference = 'average'

###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# Some directories
fname.add('raw_data_dir', raw_data_dir)
fname.add('processed_data_dir', './processed')
fname.add('figures_dir', './figures')

# The data files that are used by the analysis steps
fname.add('precomputed_ica_tsv', '{raw_data_dir}/{task}/sub-{subject:03d}/ses-{task}/eeg/sub-{subject:03d}_ses-{task}_task-{task}_ica.tsv')
fname.add('precomputed_ica_set', '{raw_data_dir}/{task}/sub-{subject:03d}/ses-{task}/eeg/sub-{subject:03d}_ses-{task}_task-{task}_ica.set')
fname.add('precomputed_badChannels', '{raw_data_dir}/{task}/sub-{subject:03d}/ses-{task}/eeg/sub-{subject:03d}_ses-{task}_task-{task}_badChannels.tsv')
fname.add('precomputed_badSegments', '{raw_data_dir}/{task}/sub-{subject:03d}/ses-{task}/eeg/sub-{subject:03d}_ses-{task}_task-{task}_badSegments.csv')

# The data files that are produced by the analysis steps
fname.add('filtering', '{processed_data_dir}/filtered-{subject}.fif')
fname.add('epoching', '{processed_data_dir}/epochs-{subject}.fif')
fname.add('cleaned_epochs', '{processed_data_dir}/cleaned_epochs.fif')

# The figures
fname.add('figure1', '{figures_dir}/figure1.pdf')
fname.add('figure_grand_average', '{figures_dir}/figure_grand_average.pdf')

# Filenames for MNE reports
fname.add('reports_dir', './reports/')
fname.add('report', '{reports_dir}/{subject}-report.h5')
fname.add('report_html', '{reports_dir}/{subject}-report.html')

# File produced by check_system.py
fname.add('system_check', './system_check.txt')
