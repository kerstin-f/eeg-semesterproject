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
from config import fname, conditions

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# The evoked data sets are created by averaging different conditions.
epochs = mne.read_epochs(fname.cleaned_epochs(subject=subject), preload=True)

evokeds = []
# Problem: How to cope with condition names stimulus11, stimulus212
# for condition in conditions:
#     evoked = epochs[condition].average()
#     evokeds[condition] = evoked

evokeds.append(epochs.average())

# Missing: if cfg.contrasts

mne.write_evokeds(fname.evokeds(subject=subject), evokeds, overwrite=True)

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    for evoked in evokeds:
        fig = evoked.plot(show=False)
        report.add_figure(fig,'Evoked', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)