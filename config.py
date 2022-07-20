"""
===========
Config file
===========

Configuration parameters for the study.
"""

import os
import getpass
#from socket import getfqdn

from numpy import NaN
from fnames import FileNames

# My laptop
raw_data_dir = './data'
n_jobs = 1

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)


###############################################################################
# These are all the relevant parameters for the analysis.

bids_root = "./data/p3/"
task = "p3" 
analyze_channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'P3', 'P4']

# # All subjects from 1 to N
# count = 0
# for _, dirs, _ in os.walk(bids_root):
# 	for dir in dirs:
# 		if(dir.startswith('sub-')):
# 			count = count + 1
# subjects = ["%.3d" % i for i in range(1,count+1)]

subjects = ['002']


###############################################################################
# FREQUENCY FILTERING
l_freq = 0.5
h_freq = 50.0

plot_channel_filtering = 'Pz'

###############################################################################
# EPOCHING

# time relative to each event at which to start and end each epoch
epochs_tmin = -0.1
"""
The beginning of the time window for metadata generation, in seconds,
relative to the time-locked event of the respective epoch. This may be less
than or larger than the epoch's first time point. If ``None``, use the first
time point of the epoch.
"""
epochs_tmax = 1
"""
Same as ``epochs_metadata_tmin``, but specifying the **end** of the time
window for metadata generation.
"""

plot_channel_epoching = 'Pz'

###############################################################################
# DECODING

# Task P3
targets = ['stimulus:11', 'stimulus:22', 'stimulus:33', 'stimulus:44', 'stimulus:55']
distractors = ['stimulus:21', 'stimulus:31', 'stimulus:41', 'stimulus:51', 'stimulus:12', 'stimulus:32', 'stimulus:42', 'stimulus:52', 'stimulus:13', 'stimulus:23',
               'stimulus:43', 'stimulus:53', 'stimulus:14', 'stimulus:24', 'stimulus:34', 'stimulus:54', 'stimulus:15', 'stimulus:25', 'stimulus:35', 'stimulus:45']

decoding_n_splits: int = 5
"""
The number of folds (also called "splits") to use in the cross-validation scheme.
"""

random_state = 42
"""
You can specify the seed of the random number generator (RNG).
This setting is passed to the ICA algorithm and to the decoding function,
ensuring reproducible results. Set to ``None`` to avoid setting the RNG
to a defined state.
"""

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
fname.add('events', '{raw_data_dir}/{task}/sub-{subject}/ses-{task}/eeg/sub-{subject}_ses-{task}_task-{task}_events.tsv')
fname.add('event_code_values', '{raw_data_dir}/{task}/task-{task}_events.json')
fname.add('precomputed_ica_tsv', '{raw_data_dir}/{task}/sub-{subject}/ses-{task}/eeg/sub-{subject}_ses-{task}_task-{task}_ica.tsv')
fname.add('precomputed_ica_set', '{raw_data_dir}/{task}/sub-{subject}/ses-{task}/eeg/sub-{subject}_ses-{task}_task-{task}_ica.set')
fname.add('precomputed_badChannels', '{raw_data_dir}/{task}/sub-{subject}/ses-{task}/eeg/sub-{subject}_ses-{task}_task-{task}_badChannels.tsv')
fname.add('precomputed_badSegments', '{raw_data_dir}/{task}/sub-{subject}/ses-{task}/eeg/sub-{subject}_ses-{task}_task-{task}_badSegments.csv')

# The data files that are produced by the analysis steps
fname.add('filtering', '{processed_data_dir}/{subject}-filtered-raw.fif')
fname.add('epoching', '{processed_data_dir}/{subject}-epo.fif')
fname.add('cleaned_epochs', '{processed_data_dir}/{subject}-cleaned-epo.fif')
fname.add('evokeds', '{processed_data_dir}/{subject}-ave.fif')
fname.add('decoding', '{processed_data_dir}/{subject}-decoding.fif')

# The figures
fname.add('figure1', '{figures_dir}/figure1.pdf')
fname.add('figure_grand_average', '{figures_dir}/figure_grand_average.pdf')

# Filenames for MNE reports
fname.add('reports_dir', './reports/')
fname.add('report', '{reports_dir}/{subject}-report.h5')
fname.add('report_html', '{reports_dir}/{subject}-report.html')

# File produced by check_system.py
fname.add('system_check', './system_check.txt')
