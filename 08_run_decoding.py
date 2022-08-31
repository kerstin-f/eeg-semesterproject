"""
Entire epochs are fed into a logistic regression model. Decoding performance
then tells how well the classifier could predict which epoch belongs to which condition.

The different approaches for decoding are:
- Logistic regression (sklearn) + sliding estimator (mne)
- Common spacial patters (mne) + linear discriminant analysis (sklearn) with manual time windows
- Temporal generalization: Logistic regression (sklearn) + generalizing estimator (mne)

"""

import utils
import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level) #cfg.mne_log_level) 
import numpy as np
import sklearn.linear_model
import sklearn.pipeline
import sklearn.model_selection
import sklearn.discriminant_analysis 
import sklearn.preprocessing
from matplotlib import pyplot as plt
from datetime import datetime

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# ---------------------------------------------------------------------------------------------------------

epochs = mne.read_epochs(cfg.fname.epoched(subject=subject), preload=True)
epochs.pick_types(eog=False, eeg=True, exclude='bads')

epochs.set_eeg_reference('average')

# Equalize event counts and concatenate
epochs_t = epochs['Target'].copy()
epochs_d = epochs['Distractor'].copy()
epochs_list = [epochs_t, epochs_d]
mne.epochs.equalize_epoch_counts(epochs_list)
epochs =  mne.concatenate_epochs(epochs_list)
epochs.pick_channels(cfg.roi)

# List of events types (e.g. [0 0 0 1 0 1 ...]) mapped to Distractor=0, Target=1
labels = epochs.events[:, 2]   

# Cross validation split strategies:
# StratifiedKFold for logistic regression, StratifiedShuffleSplit for CSP+LDA
cv_log_reg = sklearn.model_selection.StratifiedKFold(cfg.cv_fold)
cv_csp_lda = sklearn.model_selection.StratifiedShuffleSplit(cfg.cv_fold, test_size=0.1, random_state=42)

# ---------------------------------------------------------------------------------------------------------

print('>>> Logistic regression with sliding estimator')
pipe_log_reg = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.LogisticRegression(solver='liblinear'))
epochs_resampled = epochs.copy().resample(cfg.resample)
X = epochs_resampled.get_data()

scores_log_reg_scoring_methods = {}
# 'precision': UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
# 'f1_samples': Samplewise metrics are not available outside of multilabel classification.
# 'top_k_accuracy': UndefinedMetricWarning: 'k' (2) greater than or equal to 'n_classes' (2) will result in a perfect score and is therefore meaningless.

methods = ['roc_auc', 'accuracy', 'average_precision', 'recall']
for scoring_method in methods: #['recall', 'roc_auc', 'average_precision', 'neg_log_loss', 'accuracy', 'top_k_accuracy']:
    print('>>> Scoring:', scoring_method)

    start_time = datetime.now()
    sliding_estimator = mne.decoding.SlidingEstimator(pipe_log_reg, n_jobs=cfg.n_jobs, scoring=scoring_method,verbose=cfg.mne_log_level)
    scores_log_reg_scoring_methods[scoring_method] = mne.decoding.cross_val_multiscore(sliding_estimator, X, labels, cv=cv_log_reg, n_jobs=cfg.n_jobs, verbose=cfg.mne_log_level)
    scores_log_reg_scoring_methods[scoring_method] = np.mean(scores_log_reg_scoring_methods[scoring_method], axis=0)

    execution_time = (datetime.now() - start_time).total_seconds()                                                                      
    print('>>> Duration: {:.2f}s'.format(execution_time))

# ---------------------------------------------------------------------------------------------------------

print('>>> Resample data as in temporal generalization for comparison')
epochs_resampled_log_reg_compare = epochs.copy().resample(cfg.resample_temp_gen)
X = epochs_resampled_log_reg_compare.get_data()
sliding_estimator = mne.decoding.SlidingEstimator(pipe_log_reg, n_jobs=cfg.n_jobs, scoring='roc_auc', verbose=cfg.mne_log_level)
scores_log_reg_compare = mne.decoding.cross_val_multiscore(sliding_estimator, X, labels, cv=cv_log_reg, n_jobs=cfg.n_jobs, verbose=cfg.mne_log_level)
scores_log_reg_compare = np.mean(scores_log_reg_compare, axis=0)

# ---------------------------------------------------------------------------------------------------------

print('>>> Common spacial patterns + linear discriminant analysis with manual time windows')
start_time = datetime.now()
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
csp = mne.decoding.CSP(n_components=2, reg=cfg.reg_csp)
pipe_csp_lda = sklearn.pipeline.Pipeline([('CSP', csp), ('LDA', lda)])
w_size = 0.1
times_lda_all = epochs.times
times_lda_all = times_lda_all[::10]
times_lda = []
scores_lda = []

# 2 Hz high-pass helps improve CSP
epochs_filtered = epochs.copy().filter(l_freq=2.0,h_freq=cfg.high_freq, fir_design='firwin', phase='zero')

for i, w_time in enumerate(times_lda_all):
        print("{}/{}".format(i, len(times_lda_all)))
        w_tmin = w_time - w_size / 2.
        w_tmax = w_time + w_size / 2.
        if w_tmin < times_lda_all[0]:
            continue
        if w_tmax > times_lda_all[len(times_lda_all)-1]:
            continue

        times_lda.append(w_time)
        X_cropped = epochs_filtered.copy().crop(w_tmin, w_tmax).get_data()

        score = sklearn.model_selection.cross_val_score(estimator=pipe_csp_lda, X=X_cropped, y=labels,
                                                        scoring=cfg.scoring, cv=cv_csp_lda,
                                                        n_jobs=cfg.n_jobs, error_score='raise', verbose=1)
        scores_lda.append(score)

scores_lda = np.mean(scores_lda, axis=1)
execution_time_lda = (datetime.now() - start_time).total_seconds()                                                                             
print('>>> Duration: {:.2f}s'.format(execution_time_lda))

# ---------------------------------------------------------------------------------------------------------

print('>>> Temporal generalization')
start_time = datetime.now()
pipe_log_reg = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.LogisticRegression(solver='liblinear'))
epochs_resampled_temp_gen = epochs.copy().resample(cfg.resample_temp_gen)
X = epochs_resampled_temp_gen.get_data()
generalizing_estimator = mne.decoding.GeneralizingEstimator(pipe_log_reg, n_jobs=cfg.n_jobs, scoring=cfg.scoring, verbose=cfg.mne_log_level)
scores_temp_gen = mne.decoding.cross_val_multiscore(generalizing_estimator, X, labels, cv=cv_log_reg, n_jobs=cfg.n_jobs, verbose=cfg.mne_log_level)
scores_temp_gen = np.mean(scores_temp_gen, axis=0)
execution_time_temp_gen = (datetime.now() - start_time).total_seconds()                                                                             
print('>>> Duration: {:.2f}s'.format(execution_time_temp_gen))

# ---------------------------------------------------------------------------------------------------------

print('>>> Equality check of SlidingEstimator and GeneralizingEstimator')
np.testing.assert_almost_equal(scores_log_reg_compare, np.diag(scores_temp_gen), err_msg='Results of decoding over time differs for SlidingEstimator and GeneralizingEstimator')


if cfg.generate_reports:
    # Add a plot of the data to the HTML report
    with mne.open_report(cfg.fname.report_decoding(subject=subject)) as report:
        # ---------------------------------------------------------------------------------------------------------
        print('>>> Plots')
        report.title = 'Decoding: Subject ' + subject

        # ---------------------------------------------------------------------------------------------------------
        # Fit CSP on epochs to plot filters and patterns
        csp.fit(epochs_filtered.get_data(), labels)
        fig_csp_patterns = csp.plot_patterns(epochs.info)
        fig_csp_filters = csp.plot_filters(epochs.info, scalings=1e-4)

        # ---------------------------------------------------------------------------------------------------------
        ymin = 0.25
        ymax = 1.0

        xmin = cfg.epochs_tmin
        xmax = cfg.epochs_tmax
        
        fig_scores_lda, ax = plt.subplots()
        ax.plot(times_lda, scores_lda, label='Score')
        ax.axhline(.5, color='k', linestyle='--', label='Chance')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('AUC')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.legend()
        ax.axvline(.0, color='k', linestyle='-')
        # ---------------------------------------------------------------------------------------------------------
        fig_scores_log_reg, ax = plt.subplots()
        ax.plot(epochs_resampled.times, scores_log_reg_scoring_methods['roc_auc'], label='Score')
        ax.axhline(.5, color='k', linestyle='--', label='Chance')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('AUC')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.legend()
        ax.axvline(.0, color='k', linestyle='-')      

        # ---------------------------------------------------------------------------------------------------------
  
        fig_matrix_temp_gen, ax = plt.subplots(1, 1)
        im = ax.imshow(scores_temp_gen, interpolation='lanczos', origin='lower', cmap='RdBu_r',
                    extent=epochs_resampled_temp_gen.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
        ax.set_xlabel('Testing Time (s)')
        ax.set_ylabel('Training Time (s)')
        ax.axvline(0, color='k')
        ax.axhline(0, color='k')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('AUC')

        # ---------------------------------------------------------------------------------------------------------

        utils.add_html(report, 'decoding.implementation','Task run_decoding', 'Decoding')

        utils.add_html(report, 'decoding.csp','DESC: CSP Patterns and Filters', 'Decoding')       
        title_left = 'Patterns'
        title_right = 'Filters'
        fig_csp_combined = utils.combine_figures(fig_csp_patterns, title_left, fig_csp_filters, title_right, fig_size=(14,4))
        report.add_figure(fig_csp_combined, 'CSP Patterns and Filters', replace=True, tags='Decoding')

        utils.add_html(report, 'decoding.decoding_over_time','DESC: Decoding over time', 'Decoding')
        title_left = 'CSP + LinearDiscriminantAnalysis'
        title_right = 'LogisticRegression, scoring=roc_auc'
        fig_decoding_combined = utils.combine_figures(fig_scores_lda, title_left, fig_scores_log_reg, title_right, fig_size=(16,7))
        report.add_figure(fig_decoding_combined, 'Decoding over time', replace=True, tags='Decoding')

        # # ---------------------------------------------------------------------------------------------------------
        # plt.figure()
        # num_plt_rows = 2
        # num_plt_cols = 2
        # fig_scoring_methods, ax_arr = plt.subplots(num_plt_rows, num_plt_cols, figsize=(num_plt_cols*5, num_plt_rows*5), layout='tight')
        # plt.subplots_adjust(wspace=0, hspace=0)
        # plt.autoscale()
        # methods = list(scores_log_reg_scoring_methods.keys())
        # n = 0
        # for i in range(0,num_plt_rows):
        #     for j in range(0,num_plt_cols):
        #         ax = ax_arr[i,j]
        #         if n >= len(methods):
        #             break
        #         scoring_method = methods[n]
        #         n+=1
        #         ax.plot(epochs_resampled.times, scores_log_reg_scoring_methods[scoring_method], label=scoring_method)
        #         ax.set_xlim([xmin, xmax])
        #         ax.set_ylim([ymin, ymax])
        #         ax.set_xlabel('time (s)')
        #         #ax.set_ylabel(scoring_method)
        #         ax.legend(loc='upper left')
        #         # ax.axvline(.0, color='k', linestyle='dashed')
        #         #title = scoring_method
        #         #ax.title.set_text(title)    

        # report.add_figure(fig_scoring_methods, 'Comparison of scoring methods', replace=True, tags='Decoding')

        utils.add_html(report, 'decoding.temporal_generalization','DESC: Temporal Generalization Matrix', 'Decoding')
        report.add_figure(fig_matrix_temp_gen, 'Temporal Generalization Matrix, resampling={}'.format(cfg.resample_temp_gen), replace=True, tags='Decoding')     

        utils.add_html(report, 'decoding.compare_diagonal','Relationship between decoding over time and temporal generalization', 'Decoding')

        report.save(cfg.fname.report_decoding_html(subject=subject), overwrite=True, open_browser=False)

print('>>> Done')
