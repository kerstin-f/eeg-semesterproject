"""
Perform bandpass filtering.
"""
import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level) 
import numpy as np
import pandas as pd
import utils
from mne.preprocessing import read_ica
from matplotlib import pyplot as plt

# All parameters are defined in config.py

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Missing: Adding report of ica

# Load ICA.
ica, badComps = utils.load_precomputed_ica(subject)

# Problem: Wieso hier raw und nicht epochs?
raw = mne.io.read_raw_fif(cfg.fname.filtered(subject=subject), preload=True)
ica = utils.add_ica_info(raw, ica)

# Select ICs to remove.
ica.exclude = badComps

# Load epochs to reject ICA components
epochs = mne.read_epochs(cfg.fname.epoched(subject=subject), preload=True)

# Missing: epochs.drop_bad(cfg.ica_reject)

cleaned_epochs = ica.apply(epochs.copy())

# Save reconstructed epochs after ICA
cleaned_epochs.save(cfg.fname.epoched_cleaned(subject=subject), overwrite=True)

# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report(subject=subject)) as report:
    # Missing: Plot raw data and power spectral density.
    fig1 = ica.plot_components(range(20), show=False)
    report.add_figure(fig1,'ICA components')

    # Compare ERP/ERF before and after ICA artifact rejection. The evoked
    # response is calculated across ALL epochs, just like ICA was run on
    # all epochs, regardless of their respective experimental condition.
    #
    # We apply baseline correction here to (hopefully!) make the effects of
    # ICA easier to see. Otherwise, individual channels might just have
    # arbitrary DC shifts, and we wouldn't be able to easily decipher what's
    # going on!

    # Plot an overlay of the original signal against the reconstructed signal with the artifactual ICs excluded
    # Problem: Was sind die artifactual components hier?
    fig2 = ica.plot_overlay(raw,exclude=[1,8,9], show=False)
    report.add_figure(fig2,'Signals before (red) and after (black) cleaning raw data')

    # Problem: Was mit Baseline machen?
    # report.add_ica(ica=ica, title='Effects of ICA cleaning', inst=epochs.copy().apply_baseline('average'))

    # Problem: Does add_ica_info fix the problem?
    # Inspect a specific component
    # fig3 = ica.plot_properties(raw,picks=cfg.plot_channel_epoching,psd_args={'fmax': 35.},reject=None, show=False);
    # report.add_figure(fig3,'ICA properties of channel ' + cfg.plot_channel_epoching)

    report.save(cfg.fname.report_html(subject=subject), overwrite=True, open_browser=False)