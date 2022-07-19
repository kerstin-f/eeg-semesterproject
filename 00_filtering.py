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
import config as cfg

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)


# Load the data, filter it and save the result
raw = utils.import_raw(subject, cfg.task)
# raw.load_data()

if cfg.l_freq is not None and cfg.h_freq is None:
    msg = (f'High-pass filtering EEG data; lower bound: 'f'{cfg.l_freq} Hz')
elif cfg.l_freq is None and cfg.h_freq is not None:
    msg = (f'Low-pass filtering EEG data; upper bound: 'f'{cfg.h_freq} Hz')
elif cfg.l_freq is not None and cfg.h_freq is not None:
    msg = (f'Band-pass filtering EEG data; range: 'f'{cfg.l_freq} – {cfg.h_freq} Hz')
else:
    msg = (f'Not applying frequency filtering to EEG data.')

raw_filtered = raw.copy().filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq,fir_design='firwin')

# Account for precomputed bad data
# Missing: Report badData
# https://mne.tools/dev/auto_tutorials/preprocessing/15_handling_bad_channels.html
badAnnotations, badChannels = utils.load_precomputed_badData(subject)
raw_filtered.annotations.append(badAnnotations.onset,badAnnotations.duration,badAnnotations.description)
raw_filtered.info['bads'].extend(badChannels)
raw_filtered.interpolate_bads()

raw_filtered.save(cfg.fname.filtering(subject=subject), overwrite=True)

# Missing: Resample (directly after filtering)
# Missing: Rereference? (=> see 01_epoching.py)

# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report(subject=subject)) as report:
    report.title = 'Results for subject ' + subject
    report.add_raw(raw=raw, title='Raw', psd=False, replace=True)

    # Plot events
    # Missing: use sampling frequency from raw.info['sfreq']?

    # df = pd.read_csv(cfg.fname.events(subject=subject, task=cfg.task), delimiter='\t')
    # stim = df[df.trial_type == 'stimulus']
    # res = df[df.trial_type == 'response']
    # fig1 = plt.figure()
    # plt.scatter(stim.onset, stim.trial_type, marker='o', c='green')
    # plt.scatter(res.onset, res.trial_type, marker='o', c='red')
    # report.add_figure(fig1,'Events from events.tsv', replace=True)

    # Plots for the plot_channel set in config file
    raw_subselect = raw.copy().pick_channels([cfg.plot_channel_filtering])
    raw_filtered_subselect = raw_filtered.copy().pick_channels([cfg.plot_channel_filtering])

    # Plot raw: extract a single channel and plot the whole timeseries
    # Problem: fehlender Plot??
    fig2 = plt.figure()
    plt.plot(raw=raw_subselect[:,:][0].T, show=False)
    report.add_figure(fig2,'Whole timeseries of channel '+ cfg.plot_channel_filtering, replace=True)
    
    # Plot raw psd
    fig3 = raw_subselect.plot_psd(area_mode='range', tmax=10.0, average=False, xscale="linear", show=False)
    report.add_figure(fig3,'Power spectral density of channel ' + cfg.plot_channel_filtering + ' (xscale = linear)', replace=True)
    
    # Plot raw_filtered psd
    fig4 = raw_filtered_subselect.plot_psd(area_mode='range', tmax=10.0, average=False, xscale="log", show=False)
    report.add_figure(fig4,'Power spectral density of filtered channel ' + cfg.plot_channel_filtering + ' (xscale = log)', replace=True)

    # Compare raw and raw_filtered
    fig4 = plt.figure()
    plt.plot(raw_subselect[:,0:1000][0].T-np.median(raw_subselect[:,0:1000][0].T), label="raw")
    plt.plot(raw_filtered_subselect[:,0:1000][0].T-np.median(raw_filtered_subselect[:,0:1000][0].T), label="filtered")
    plt.legend(loc="upper left")
    report.add_figure(fig4,'Comparison of raw and filtered channel ' + cfg.plot_channel_filtering, replace=True)

    report.save(cfg.fname.report_html(subject=subject), overwrite=True, open_browser=False)
