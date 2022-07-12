"""
Perform bandpass filtering.
"""
import argparse
import mne
import utils
import numpy as np
from mne_bids import (BIDSPath, read_raw_bids)
from matplotlib import pyplot as plt

# All parameters are defined in config.py
from config import fname, task, l_freq, h_freq, h_trans_bandwidth, l_trans_bandwidth

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Load the data, filter it and save the result
raw = utils.import_data(subject, task)
# raw.load_data()

if l_freq is not None and h_freq is None:
    msg = (f'High-pass filtering EEG data; lower bound: '
        f'{l_freq} Hz')
elif l_freq is None and h_freq is not None:
    msg = (f'Low-pass filtering EEG data; upper bound: 'f'{h_freq} Hz')
elif l_freq is not None and h_freq is not None:
    msg = (f'Band-pass filtering EEG data; range: 'f'{l_freq} – {h_freq} Hz')
else:
    msg = (f'Not applying frequency filtering to EEG data.')

# Problem: introduce optional arguments
if l_trans_bandwidth is not None and h_trans_bandwidth is not None:
    raw.filter(l_freq=l_freq, h_freq=h_freq,
               l_trans_bandwidth=l_trans_bandwidth,
               h_trans_bandwidth=h_trans_bandwidth,
               filter_length='auto', phase='zero', fir_window='hamming',
               fir_design='firwin', n_jobs=1)
elif l_freq is not None or h_freq is not None:
    raw.filter(l_freq=l_freq, h_freq=h_freq,
               filter_length='auto', phase='zero', fir_window='hamming',
               fir_design='firwin', n_jobs=1)

raw.save(fname.filtering(subject=subject), overwrite=True)

# Missing: Resample (directly after filtering)

# Missing: Rereference? (=> see 01_epoching.py)

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    # Missing: Plot raw data and power spectral density.
    fig = plt.figure()
    plt.plot(raw=raw.copy().pick_channels(["Pz"])[:,:][0].T,show=False)
    # report.add_figs_to_section(fig, 'Filtered data', section='filtering', replace=True)
    report.add_figure(fig,'Filtered data: Channel Pz')
    report.save(fname.report_html(subject=subject), overwrite=True,
                open_browser=False)