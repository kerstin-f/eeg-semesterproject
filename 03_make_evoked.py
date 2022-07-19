"""
Perform bandpass filtering.
"""
import argparse
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import config as cfg

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

epochs = mne.read_epochs(cfg.fname.cleaned_epochs(subject=subject), preload=True)

# The evoked data sets are created by averaging different conditions
all_evoked = dict()
for e in epochs.event_id.keys():
    evoked = epochs[e].average()
    all_evoked[e] = evoked

evokeds = list(all_evoked.values())
mne.write_evokeds(cfg.fname.evokeds(subject=subject), evokeds, overwrite=True)

# Problem: How to cope with condition names stimulus11, stimulus12..

# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report(subject=subject)) as report:
    # Missing: add evokeds information to report
    # HERE: e.g. for subset of evokeds
    # report.add_evokeds(evokeds=evokeds)
    for e in ['stimulus:11', 'stimulus:14']:
        fig1 = all_evoked[e].plot(show=False, ylim=dict(eeg=(-20, 20)), spatial_colors=True)
        report.add_figure(fig1,'Evoked ' + e, replace=True)
   
    # Compare ERPs of targets and distractors
    fig2 = mne.viz.plot_compare_evokeds({'Target (stimulus:11)': all_evoked['stimulus:11'],
                                            'Distractor (stimulus:14)': all_evoked['stimulus:14']},
                                            picks=cfg.plot_channel_epoching,
                                            show=False)
    report.add_figure(fig2,'Comparison of ERPs (target - distractor) of '+ cfg.plot_channel_epoching, replace=True)
    report.save(cfg.fname.report_html(subject=subject), overwrite=True, open_browser=False)
