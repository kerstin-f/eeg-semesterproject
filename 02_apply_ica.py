"""
Perform bandpass filtering.
"""
import argparse
import mne
import numpy as np
import pandas as pd
import utils
from mne_bids import (BIDSPath, read_raw_bids)
from mne.preprocessing import read_ica
from matplotlib import pyplot as plt
from config import fname

# All parameters are defined in config.py
from config import fname, task

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Missing: Adding report of ica

# Load ICA.
ica, badComps = utils.load_precomputed_ica(subject)

# Problem: Wiese hier raw und nicht epochs?
ica = utils.add_ica_info(utils.import_raw(subject, task), ica)

# Select ICs to remove.
ica.exclude = badComps

# Load epochs to reject ICA components.
epochs = mne.read_epochs(fname.epoching(subject=subject), preload=True)

# Missing: epochs.drop_bad(cfg.ica_reject)

cleaned_epochs = ica.apply(epochs.copy())

# Save reconstructed epochs after ICA
cleaned_epochs.save(fname.cleaned_epochs(subject=subject), overwrite=True)

    # # Compare ERP/ERF before and after ICA artifact rejection. The evoked
    # # response is calculated across ALL epochs, just like ICA was run on
    # # all epochs, regardless of their respective experimental condition.
    # #
    # # We apply baseline correction here to (hopefully!) make the effects of
    # # ICA easier to see. Otherwise, individual channels might just have
    # # arbitrary DC shifts, and we wouldn't be able to easily decipher what's
    # # going on!
    # report = Report(report_fname, title=title, verbose=False)
    # picks = ica.exclude if ica.exclude else None
    # report.add_ica(
    #     ica=ica,
    #     title='Effects of ICA cleaning',
    #     inst=epochs.copy().apply_baseline(cfg.baseline),
    #     picks=picks
    # )
    # report.save(report_fname, overwrite=True, open_browser=cfg.interactive)

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    # Missing: Plot raw data and power spectral density.
    # fig = mne.viz.plot_evoked(evoked=epochs.average(), show=False, picks="Cz")
    fig_comp = ica.plot_components(range(20), show=False)
    report.add_figure(fig_comp,'ICA components:')
    # Problem: Does add_ica_info fix the problem?
    # fig_props = ica.plot_properties(epochs,show=False,psd_args={'fmax': 35.},reject=None)
    # report.add_figure(fig_comp,'ICA properties: Channel Cz:')
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)