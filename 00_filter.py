"""
Perform bandpass filtering.
"""
import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level) 
import utils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal

# All parameters are defined in config.py

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Load the data, filter it and save the result
raw = utils.import_raw(subject, cfg.task)

# Apply frequency filtering
raw_filtered_acausal = raw.copy().filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq,fir_design='firwin', phase='zero', skip_by_annotation='edge')
raw_filtered_causal = raw.copy().filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq,fir_design='firwin', phase='minimum', skip_by_annotation='edge')

raw_filtered_acausal.save(cfg.fname.filtered(subject=subject), overwrite=True)

# Missing: Rereference? (=> see 01_make_epochs.py)
print("test")
# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report(subject=subject)) as report:
    report.title = 'Results for subject ' + subject
    report.add_raw(raw=raw, title='Raw', psd=False, butterfly=False, replace=True)

    # Plot impulse response function
    fig1 = plt.figure()
    imp = signal.unit_impulse(100, 'mid')
    impulse_response = mne.filter.filter_data(imp,sfreq=raw.info['sfreq'],l_freq=cfg.l_freq, h_freq=cfg.h_freq,fir_design='firwin')
    plt.plot(np.arange(-50, 50), imp)
    plt.plot(np.arange(-50, 50), impulse_response)
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    report.add_figure(fig1,'FIR filter impulse response function', replace=True)

    # Subselect channel of interest based on setting in config file
    raw_subselect = raw.copy().pick_channels([cfg.plot_channel_filtering])
    raw_filtered_subselect_acausal = raw_filtered_acausal.copy().pick_channels([cfg.plot_channel_filtering])
    raw_filtered_subselect_causal = raw_filtered_causal.copy().pick_channels([cfg.plot_channel_filtering])
 
    # Plot raw psd
    fig3 = raw_subselect.plot_psd(area_mode='range', tmax=10.0, average=False, xscale="linear", show=False)
    report.add_figure(fig3,'Power spectral density of channel ' + cfg.plot_channel_filtering + ' (xscale = linear)', replace=True)
    
    # Plot raw_filtered psd
    fig4 = raw_filtered_subselect_acausal.plot_psd(area_mode='range', tmax=10.0, average=False, xscale="log", show=False)
    report.add_figure(fig4,'Power spectral density of filtered channel ' + cfg.plot_channel_filtering + ' (xscale = log)', replace=True)

    # Compare raw and raw_filtered
    fig4 = plt.figure()
    plt.plot(raw_subselect[:,0:1000][0].T-np.median(raw_subselect[:,0:1000][0].T), label="raw")
    plt.plot(raw_filtered_subselect_acausal[:,0:1000][0].T-np.mean(raw_filtered_subselect_acausal[:,0:1000][0].T), label="filtered")
    plt.legend(loc="upper left")
    report.add_figure(fig4,'Comparison of raw and filtered channel ' + cfg.plot_channel_filtering, replace=True)

    # Compare raw filtered by acausal and causal filters
    fig5 = plt.figure()
    plt.plot(raw_subselect[:,0:1000][0].T-np.median(raw_subselect[:,0:1000][0].T), label="raw")
    plt.plot(raw_filtered_subselect_acausal[:,0:1000][0].T-np.median(raw_filtered_subselect_acausal[:,0:1000][0].T), label="acausal")
    plt.plot(raw_filtered_subselect_causal[:,0:1000][0].T-np.mean(raw_filtered_subselect_causal[:,0:1000][0].T), label="causal")
    plt.legend(loc="upper left")
    report.add_figure(fig5,'Comparison of acausal and causal filtering result of filtered channel ' + cfg.plot_channel_filtering, replace=True)

    report.save(cfg.fname.report_html(subject=subject), overwrite=True, open_browser=False)
