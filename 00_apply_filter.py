"""
Perform band-pass filtering on raw data.
"""
import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level) 
import utils
import numpy as np
from matplotlib import pyplot as plt

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Load raw data of the subject to be processed
raw = utils.import_raw(subject, cfg.task)

assert 0.0 < cfg.low_freq < 1, 'The low-frequency range should be between 0 and 1.'
assert 20 < cfg.high_freq < 70, 'The high-frequency range should be between 20 and 70.'
assert isinstance(cfg.choi, str) and cfg.choi in raw.ch_names, 'Channel to be plottet not in raw.'

# Apply frequency filtering
raw_filtered_acausal = raw.copy().filter(l_freq=cfg.low_freq, h_freq=cfg.high_freq,fir_design='firwin', phase='zero')
raw_filtered_causal = raw.copy().filter(l_freq=cfg.low_freq, h_freq=cfg.high_freq,fir_design='firwin', phase='minimum')

# Save filtered raw
raw_filtered_acausal.save(cfg.fname.filtered(subject=subject), overwrite=True)

# Subselect channel of interest for plotting
raw_subselect = raw.copy().pick_channels([cfg.choi])
raw_subselect_acausal = raw_filtered_acausal.copy().pick_channels([cfg.choi])
raw_subselect_causal = raw_filtered_causal.copy().pick_channels([cfg.choi])

# Add a plot of the data to the HTML report
if cfg.generate_reports:
    with mne.open_report(cfg.fname.report_preprocessing(subject=subject)) as report:
        report.title = 'Preprocessing: Subject ' + subject
        # ---------------------------------------------------------------------------------------------------------
        # Plot psd of raw and filtered raw
        kwargs = dict(area_mode='range', tmax=10.0, average=False, fmax=100, show=False)
        fig_psd_original = raw_subselect.plot_psd(xscale="linear", **kwargs)
        fig_psd_filtered = raw_subselect_acausal.plot_psd(xscale="log", **kwargs)
        
        # ---------------------------------------------------------------------------------------------------------
        # Compare raw and raw_filtered
        fig_comparison = plt.figure()
        plt.plot(raw_subselect[:,0:1000][0].T-np.median(raw_subselect[:,0:1000][0].T), label="raw")
        plt.plot(raw_subselect_acausal[:,0:1000][0].T-np.mean(raw_subselect_acausal[:,0:1000][0].T), label="acausal")
        plt.legend(loc="upper left")

        # ---------------------------------------------------------------------------------------------------------
        # Compare raw filtered by acausal and causal filters
        fig_comparison_filters = plt.figure()
        plt.plot(raw_subselect[:,0:1000][0].T-np.median(raw_subselect[:,0:1000][0].T), label="raw")
        plt.plot(raw_subselect_acausal[:,0:1000][0].T-np.median(raw_subselect_acausal[:,0:1000][0].T), label="acausal")
        plt.plot(raw_subselect_causal[:,0:1000][0].T-np.mean(raw_subselect_causal[:,0:1000][0].T), label="causal")
        plt.legend(loc="upper left")

        # ---------------------------------------------------------------------------------------------------------     
        utils.add_html(report, 'apply_filter.implementation','Task apply_filter', 'Filtering')
        report.add_figure([fig_psd_original, fig_psd_filtered], title='Power spectral density (PSD)', caption=['original (xscale = linear)', 'filtered (xscale = log)'], replace=True, tags='Filtering')
        title_left = 'Original and filtered signal'
        title_right = 'Comparison of filters'
        fig_combined_comparisons = utils.combine_figures(fig_comparison, title_left, fig_comparison_filters , title_right)
        report.add_figure(fig_combined_comparisons,'Filtering results' + cfg.choi, replace=True, tags='Filtering')
        report.save(cfg.fname.report_preprocessing_html(subject=subject), overwrite=True, open_browser=False)
