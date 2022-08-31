"""
Analyze ERP peak on single-subject level.
"""

import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level)
import numpy as np
import pandas as pd
import seaborn as sns
import utils
from matplotlib import pyplot as plt
import itertools

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# ---------------------------------------------------------------------------------------------------------
# PREPARE DATA
# Load Epochs
epochs = mne.read_epochs(cfg.fname.epoched(subject=subject))
epochs.pick_types(meg=False, eeg=True, eog=False, exclude='bads')

# Create single-channel Epochs object 
epochs_choi = epochs.copy().pick_channels([cfg.choi])
# ICA may introduce DC offsets, i.e. baseline-correct data after cleaning
epochs_choi.apply_baseline(cfg.baseline)

# Load Evoked Diff
evoked_diff = mne.read_evokeds(cfg.fname.evokeds(subject=subject), condition='Difference')
evoked_diff_choi = evoked_diff.copy().pick_channels([cfg.choi])

# ---------------------------------------------------------------------------------------------------------
# AMPLITUDE AND LATENCY DISTRIBUTION ACROSS EPOCHS (SINGLE CHANNEL)
print('>>> Create dataframe from epochs data')
df_epochs = epochs_choi.copy().crop(cfg.test_tmin, cfg.test_tmax).to_data_frame()
# Extract maximum amplitude and corresponding latency per epoch in time window of interest
ix = df_epochs.groupby(['epoch'])[cfg.choi].transform(max) == df_epochs[cfg.choi]
df_max = df_epochs[ix]

# ---------------------------------------------------------------------------------------------------------
# PERMUTATION T-TEST (ALL CHANNELS)
# Compute mean per epoch
print('>>> Perform permutation T-test')
permutation_t_test_data = np.mean(epochs.get_data(tmin=cfg.test_tmin, tmax=cfg.test_tmax), axis=2)
info_obj = evoked_diff.copy().pick_types(meg=False, eeg=True, eog=False)
picks = mne.pick_types(info_obj.info, meg=False, eeg=True, eog=False, exclude='bads')
fig_evoked_topomap = None
fig_histogram_t_test = None
try:
    fig_evoked_topomap, fig_histogram_t_test = utils.perform_t_test(permutation_t_test_data, info_obj, picks)
except:
    print('Warning: Utils-method perform_t_test failed.')

# ---------------------------------------------------------------------------------------------------------
# PERMUTATION CLUSTER TEST (SINGLE CHANNEL)
# Extract  channel data separately per condition
print('>>> Perform permutation cluster test')
permutation_cluster_test_data = [epochs_choi[condition].get_data()[:,0,:] for condition in cfg.conditions]
fig_permutation_cluster_test = utils.perform_permutation_cluster_test(permutation_cluster_test_data, evoked_diff_choi)

del epochs
del epochs_choi

if cfg.generate_reports:
    # Add a plot of the data to the HTML report
    with mne.open_report(cfg.fname.report_analysis_single(subject=subject)) as report:
        report.title = 'Peak Analysis: Subject ' + subject
        # ---------------------------------------------------------------------------------------------------------
        html_peak_info_all, fig_peak_info_all = utils.get_peak_info(evoked_diff)
        html_peak_info_choi, fig_peak_info_choi = utils.get_peak_info(evoked_diff_choi)

        # ---------------------------------------------------------------------------------------------------------
        fig_histogram_matrix, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
        
        palette = itertools.cycle(cfg.palette)
        c = next(palette)
        sns.histplot(df_max, x=cfg.choi, alpha = .7, hue='condition', bins=30, binrange=(0,80), hue_order=[cfg.conditions[1], cfg.conditions[0]], ax=ax[0,0], color=c)
        ax[0,0].set_title('Peak Amplitudes')
        sns.histplot(df_max, x='time', alpha = .7, hue='condition', bins=30, binrange=(0,0.8), hue_order=[cfg.conditions[1], cfg.conditions[0]], ax=ax[0,1], color=c)
        ax[0,1].set_title('Peak Latencies')
        sns.histplot(df_max, x=cfg.choi, y='time', bins=30, cbar=True, discrete=(False,False), ax=ax[1,0])
        ax[1,0].set_title('Bivariate histogram of peaks and latencies')
        sns.ecdfplot(df_max, x=cfg.choi, hue='condition', ax=ax[1,1], color=c, hue_order=[cfg.conditions[1], cfg.conditions[0]])
        ax[1,1].set_title('Empirical cumulative distribution of peak amplitudes')
      
        # ---------------------------------------------------------------------------------------------------------
        utils.add_html(report, 'analyze_single.implementation','Task analyze_single', 'Analysis_Single')

        utils.add_html(report, 'analyze_single.peak_measures','DESC: Peak Measures', 'Analysis_Single')
        title_left = 'get_peak: all channels'
        title_right = 'get_peak: choi' 
        fig_peak_measures = utils.combine_figures(fig_peak_info_all, title_left, fig_peak_info_choi, title_right, fig_size=(14, 6))
        report.add_figure(fig_peak_measures,'Peak measures', replace=True, tags='Analysis_Single')
        report.add_html(html_peak_info_all,'Peak info: all channels', replace=True, tags='Analysis_Single')
        report.add_html(html_peak_info_choi,'Peak info: {}'.format(cfg.choi), replace=True, tags='Analysis_Single')

        utils.add_html(report, 'analyze_single.histograms','DESC: Amplitudes and Latencies across Epochs', 'Analysis_Single')
        report.add_figure(fig_histogram_matrix, 'Amplitudes and Latencies across Epochs', replace=True, tags='Analysis_Single')  

        utils.add_html(report, 'analyze_single.permutation_t_test','DESC: Permutation T-Test', 'Analysis_Single')
        if fig_evoked_topomap and fig_histogram_t_test:
            fig_permutation_t_test = utils.combine_figures(fig_evoked_topomap, 'Statistical Topomap', fig_histogram_t_test, 'Histogram', fig_size=(16, 6))
            report.add_figure(fig_permutation_t_test, 'Permutation T-Test', replace=True, tags='Analysis_Single')        

        utils.add_html(report, 'analyze_single.permutation_cluster_test','DESC: Permutation Cluster Test {}'.format(cfg.choi), 'Analysis_Single')
        report.add_figure(fig_permutation_cluster_test,'Permutation Cluster Test {}'.format(cfg.choi), replace=True, tags='Analysis_Single')
        
        report.save(cfg.fname.report_analysis_single_html(subject=subject), overwrite=True, open_browser=False)

