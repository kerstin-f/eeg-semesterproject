""""
Entire epochs are fed into a logistic regression model. Decoding performance
then tells how well the classifier could predict which epoch belongs to
which condition.
"""

import config as cfg
import argparse
import mne
mne.set_log_level('INFO') #cfg.mne_log_level) 
import numpy as np
#import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import sklearn.pipeline
import sklearn.model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mne.decoding import Scaler, Vectorizer
from matplotlib import pyplot as plt



# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject

print('Processing subject:', subject)
print('Contrasting conditions: Target - Distractor')

# The evoked data sets are created by averaging different conditions.
epochs = mne.read_epochs(cfg.fname.epoched_cleaned(subject=subject), preload=True)
print(' ===== CHANNELS =====')
print(epochs.ch_names)
print(' ===== END CHANNELS =====')

#['FP1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Oz', 'Pz', 'CPz',
#    'FP2', 'Fz', 'F4', 'F8', 'FC4', 'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'P10', 'PO8', 'PO4', 'O2']


ch_test = ['FP1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Oz', 'Pz',
           'CPz', 'FP2', 'Fz', 'F4', 'F8', 'FC4', 'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'P10', 'PO8', 'PO4']

#epochs.pick_channels(cfg.channels_to_analyze)
epochs.pick_channels(ch_test)

# Problem: When to pick channels? Not here...
# epochs.pick_channels(cfg.channels_to_analyze)

# We special-case the average reference here to work around a situation
# where e.g. `channels_to_analyze` might contain only a single channel:
# `concatenate_epochs` below will then fail when trying to create /
# apply the projection. We can avoid this by removing an existing    
# average reference projection here, and applying the average reference    
# directly – without going through a projector.


# Problem: in exercise epochs.copy().crop(tmin=1., tmax=2.)
# -> Avoid classification of active subject reactions 
# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.

epochs_targets = epochs['Targets']
epochs_distractors = epochs['Distractors']

# Problem: crop or not?
# epochs_combined.crop(tmin=0., tmax=1.)
# -> Don't crop, csp.fit_transform needs sufficiently large time window

# csp = mne.decoding.CSP(n_components=2)
# labels =[0]*len(epochs_targets)+[1]*len(epochs_distractors)
labels = epochs.events[:,-1]

# csp.fit_transform(epochs.get_data(), labels)
# csp_data = csp.transform(epochs.get_data())

lda = LinearDiscriminantAnalysis()
csp = mne.decoding.CSP(n_components=2, reg=0.1)
pipe = sklearn.pipeline.Pipeline([('CSP', csp), ('LDA', lda)])

# cv = sklearn.model_selection.StratifiedShuffleSplit(10, test_size=0.2, random_state=1)
# epochs_train = epochs.copy().crop(tmin=0.0, tmax=1.0)
# scores = sklearn.model_selection.cross_val_score(pipe, epochs_train.get_data(), labels, cv=cv, n_jobs=1)

# print(' ===== SCORES =====')
# print(np.mean(scores))
# print(scores)
# print(' ===== END SCORES =====')


# epochs_data_picks = epochs.get_data(picks=cfg.plot_channels_scatter).mean(axis=2)
# epochs_data_transformed = csp.transform(epochs.get_data())

w_size = 0.1
resample = 100
timeVec = epochs.copy().resample(resample).times
timeVec = timeVec[::10]
t_scores = np.zeros(len(timeVec))
for t, w_time in enumerate(timeVec):
        print("{}/{}".format(t,len(timeVec)))
        # Center the min and max of the window
        w_tmin = w_time - w_size / 2.
        w_tmax = w_time + w_size / 2.

        # stop the program if the timewindow is outside of our epoch
        if w_tmin < timeVec[0]:
            continue
        if w_tmax > timeVec[len(timeVec)-1]:
            continue
        # Crop data into time-window of interest
        X = epochs.copy().resample(resample).crop(w_tmin, w_tmax).get_data()

        # Save mean scores over folds for each frequency and time window
        t_scores[t] = np.mean( sklearn.model_selection.cross_val_score(estimator=pipe, X=X, y=labels,
                                                     scoring='roc_auc', cv=2,
                                                     n_jobs=1, error_score='raise'), axis=0)




# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report(subject=subject)) as report:
    
    report.add_epochs(epochs_targets, 'Targets', psd=False )
    report.add_epochs(epochs_distractors, 'Distractors',psd=False)
    report.add_epochs(epochs, 'epochs',psd=False)
    # report.add_epochs(epochs_combined, 'epochs_combined',psd=False)


    # Compare ERPs of targets and distractors
    fig2 = mne.viz.plot_compare_evokeds({'Target': epochs_targets.average(),
                                            'Distractor': epochs_distractors.average()},
                                            picks=cfg.plot_channels_compare_evokeds,
                                            show=False)
    report.add_figure(fig2,'Comparison of concatenated ERPs (target - distractor) of ' + str(cfg.plot_channels_compare_evokeds), replace=True)

    fig_scores_over_time = plt.figure()
    plt.plot(timeVec,t_scores,'o-')
    plt.hlines(0.5,-1,4,'k')
    report.add_figure(fig_scores_over_time,'fig_scores_over_time', replace=True)

    # fig1 = plt.figure()
    # epochs_viz = epochs.get_data(picks=['C3','C4']).mean(axis=2)
    # plt.scatter(epochs_viz[:,0],epochs_viz[:,1],color=np.array(["red","green"])[labels.astype(int)])
    # report.add_figure(fig1,'Intial data for '+ cfg.plot_channel_filtering, replace=True)

    # fig3 = plt.figure()
    # plt.scatter(epochs_data_picks[:,0],epochs_data_picks[:,1],color=np.array(["red","green"])[labels])
    # report.add_figure(fig3,'Original data for '+ str(cfg.plot_channels_scatter), replace=True)

    # fig4= plt.figure()
    # plt.scatter(epochs_data_transformed[:,0],epochs_data_transformed[:,1],color=np.array(["red","green"])[labels])
    # report.add_figure(fig4,'Transformed data for '+ str(cfg.plot_channels_scatter), replace=True)

    # fig2 = plt.figure()
    # plt.plot(timeVec,t_scores,'o-')
    # plt.hlines(0.5,-1,4,'k')
    # report.add_figure(fig2,'Mean scores over folds for each frequency and time window', replace=True)
    
    report.save(cfg.fname.report_html(subject=subject), overwrite=True, open_browser=False)
