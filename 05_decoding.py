""""
Entire epochs are fed into a logistic regression model. Decoding performance
then tells how well the classifier could predict which epoch belongs to
which condition.
"""

import argparse
import mne
import numpy as np
#import pandas as pd
import config as cfg

# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import StratifiedKFold
# from sklearn.linear_model import LogisticRegression
# import sklearn.pipeline
# import sklearn.model_selection
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mne.decoding import Scaler, Vectorizer
from matplotlib import pyplot as plt

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject

print('Processing subject:', subject)
print('Contrasting conditions: Target – Distractor')

# The evoked data sets are created by averaging different conditions.
epochs = mne.read_epochs(cfg.fname.cleaned_epochs(subject=subject), preload=True)
epochs.set_eeg_reference('average')

# Problem: When to pick channels? Not here...
# epochs.pick_channels(cfg.analyze_channels)

# We special-case the average reference here to work around a situation
# where e.g. `analyze_channels` might contain only a single channel:
# `concatenate_epochs` below will then fail when trying to create /
# apply the projection. We can avoid this by removing an existing    
# average reference projection here, and applying the average reference    
# directly – without going through a projector.


# Problem: in exercise epochs.copy().crop(tmin=1., tmax=2.)
# -> Avoid classification of active subject reactions 
# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.

epochs_combined = mne.epochs.combine_event_ids(epochs, cfg.targets, {'Targets': 0}, copy=True)
epochs_combined = mne.epochs.combine_event_ids(epochs_combined, cfg.distractors, {'Distractors': 1}, copy=True)
epochs_combined = epochs_combined[['Targets','Distractors']]

epochs_targets = epochs_combined['Targets']
epochs_distractors = epochs_combined['Distractors']

# Problem: crop or not?
#epochs_combined.crop(tmin=0., tmax=1.)

X = epochs_combined.get_data()
labels =[0]*len(epochs_targets)+[1]*len(epochs_distractors)

data = epochs_combined.get_data()
print(data.shape)

csp = mne.decoding.CSP(n_components=2)
csp.fit_transform(epochs_combined.get_data(), labels)
csp_data = csp.transform(epochs_combined.get_data())

exit()
# # Eigener sliding estimator
# lda = LinearDiscriminantAnalysis()
# csp = mne.decoding.CSP(n_components=2)
# pipe = sklearn.pipeline.Pipeline([('CSP', csp), ('LDA', lda)])
# w_size = 1
# timeVec = epochs.copy().resample(40).times
# timeVec = timeVec[::10]
# t_scores = np.zeros(len(timeVec))
# for t, w_time in enumerate(timeVec):
#         print("{}/{}".format(t,len(timeVec)))
#         # Center the min and max of the window
#         w_tmin = w_time - w_size / 2.
#         w_tmax = w_time + w_size / 2.

#         # stop the program if the timewindow is outside of our epoch
#         if w_tmin < timeVec[0]:
#             continue
#         if w_tmax > timeVec[len(timeVec)-1]:
#             continue
#         # Crop data into time-window of interest
#         X = epochs.copy().resample(40).crop(w_tmin, w_tmax).get_data()

#         # Save mean scores over folds for each frequency and time window
#         t_scores[t] = np.mean( sklearn.model_selection.cross_val_score(estimator=pipe, X=X, y=labels,
#                                                      scoring='roc_auc', cv=2,
#                                                      n_jobs=1), axis=0)

# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report(subject=subject)) as report:
    
    report.add_epochs(epochs_targets, 'Targets', psd=False )
    report.add_epochs(epochs_distractors, 'Distractors',psd=False)
    report.add_epochs(epochs, 'epochs',psd=False)
    report.add_epochs(epochs_combined, 'epochs_combined',psd=False)


    # Compare ERPs of targets and distractors
    fig2 = mne.viz.plot_compare_evokeds({'Target': epochs_targets.average(),
                                            'Distractor': epochs_distractors.average()},
                                            picks=channels,
                                            show=False)
    report.add_figure(fig2,'Comparison of concatenated ERPs (target - distractor) of ' + str(channels), replace=True)

    # fig1 = plt.figure()
    # epochs_viz = epochs.get_data(picks=['C3','C4']).mean(axis=2)
    # plt.scatter(epochs_viz[:,0],epochs_viz[:,1],color=np.array(["red","green"])[labels.astype(int)])
    # report.add_figure(fig1,'Intial data for '+ cfg.plot_channel_filtering, replace=True)

    # fig2 = plt.figure()
    # plt.scatter(csp_data[:,0],csp_data[:,1],color=np.array(["red","green"])[labels.astype(int)])
    # report.add_figure(fig2,'CSP data for '+ cfg.plot_channel_filtering, replace=True)

    # fig2 = plt.figure()
    # plt.plot(timeVec,t_scores,'o-')
    # plt.hlines(0.5,-1,4,'k')
    # report.add_figure(fig2,'Mean scores over folds for each frequency and time window', replace=True)
    
    report.save(cfg.fname.report_html(subject=subject), overwrite=True, open_browser=False)
