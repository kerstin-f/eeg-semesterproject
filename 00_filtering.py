"""
Perform bandpass filtering.
"""
import argparse
import mne
import utils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# All parameters are defined in config.py
from config import fname, task, l_freq, h_freq, analyze_channel_filtering

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Load the data, filter it and save the result
raw = utils.import_raw(subject, task)
# raw.load_data()

if l_freq is not None and h_freq is None:
    msg = (f'High-pass filtering EEG data; lower bound: 'f'{l_freq} Hz')
elif l_freq is None and h_freq is not None:
    msg = (f'Low-pass filtering EEG data; upper bound: 'f'{h_freq} Hz')
elif l_freq is not None and h_freq is not None:
    msg = (f'Band-pass filtering EEG data; range: 'f'{l_freq} – {h_freq} Hz')
else:
    msg = (f'Not applying frequency filtering to EEG data.')

# Problem: introduce optional arguments
# raw_filtered = raw.copy().filter(l_freq=l_freq,h_freq=h_freq,fir_design='firwin')
raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq,fir_design='firwin')

raw.save(fname.filtering(subject=subject), overwrite=True)

# Missing: Resample (directly after filtering)

# Missing: Rereference? (=> see 01_epoching.py)

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    report.title = 'Results for subject ' + subject
    report.add_raw(raw=raw, title='Raw', psd=False, replace=True)

    # Plot events
    # Missing: use sampling frequency from raw.info['sfreq']?
    df = pd.read_csv(fname.events(subject=subject, task=task), delimiter='\t')
    stim = df[df.trial_type == 'stimulus']
    res = df[df.trial_type == 'response']
    fig1 = plt.figure()
    plt.scatter(stim.onset, stim.trial_type, marker='o', c='green')
    plt.scatter(res.onset, res.trial_type, marker='o', c='red')
    report.add_figure(fig1,'Events from events.tsv', replace=True)

    # Plots for the analyze_channel set in config file
    raw_subselect = raw.copy().pick_channels([analyze_channel_filtering])
    raw_filtered_subselect = raw_filtered.copy().pick_channels([analyze_channel_filtering])

    # Plot raw: extract a single channel and plot the whole timeseries
    # Problem: fehlender Plot??
    fig2 = plt.figure()
    plt.plot(raw=raw_subselect[:,:][0].T, show=False)
    report.add_figure(fig2,'Whole timeseries of channel '+ analyze_channel_filtering, replace=True)
    
    # Plot raw psd
    fig3 = raw_subselect.plot_psd(area_mode='range', tmax=10.0, average=False, xscale="linear", show=False)
    report.add_figure(fig3,'Power spectral density of channel ' + analyze_channel_filtering + ' (xscale = linear)', replace=True)
    
    # Plot raw_filtered psd
    fig4 = raw_filtered_subselect.plot_psd(area_mode='range', tmax=10.0, average=False, xscale="log", show=False)
    report.add_figure(fig4,'Power spectral density of filtered channel ' + analyze_channel_filtering + ' (xscale = log)', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
