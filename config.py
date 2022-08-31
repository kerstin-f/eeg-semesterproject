"""
===========
Config file
===========

Configuration parameters for processing the ERP-CORE P3 dataset.
"""

import os
from numpy import NaN
from fnames import FileNames

###############################################################################
# General Environment settings

n_jobs = 1

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)

mne_log_level = 'ERROR'

generate_reports = True
"""
If True, expand an HTML report in each step.
If False, do not generate an HTML report.
"""

add_html_snippets = True
"""
If True, add the HTML snippets from 'report_snippets.html'
"""

###############################################################################
# ERP-CORE P3 dataset

raw_data_dir = './data'
"""
The directory containing the P3 raw data downloaded from https://figshare.com/s/5dcdc5388d4b3f37296d.
"""

task = 'P3'
"""
The name of the task of the dataset to be processed.
"""

choi = 'Pz'
"""
The name of the channel to analyze during the ERP peak analysis.
Pz is the channel for which the P3 component is expected to be largest in the difference wave. 
This channel is recommended as the a-priori measurement site for research on the P3 component. 
The pipeline plots representing a single channel always refer to this channel of interest (choi).
[source 1]
"""

roi = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'P3', 'P4']
"""
The region of interest (roi) that includes a set of channels where the P3 is expected
so that the pipeline can plot the ERP waveform averaged over these channels.
"""

###############################################################################
# General execution settings

subjects = 'all'

"""
The subjects to be processed.
One or more subjects may be specified, using the following syntax:
- List[str]: List of the numbers of the subjects to be processed (e.g. ['001'] or ['001', '008'])
- int:       Number indicating how many subjects are to be processed (starting from the first subject '001') (e.g. 3)
- 'all':     All subjects whose data can be found in raw_data_dir.

"""

subjects_manual = []
"""
The subjects for which the complete preprocessing is to be performed via the pipeline.
Attention: These subjects are only processed if they are also contained in 'subjects'.
The parameter value can be set in the same way as indicated in the comment of 'subjects'.
Manually determined cleaning times, bad channels and bad components are only provided 
for subjects ['002','003','004']. 
If 'manual_subjects' contains another subject than ['002','003','004']:
- Cleaning times and bad channels are taken from given files of P3 dataset
- ICA decomposition is explicitly calculated in run_ica.py
- Bad components are automatically detected via EOG signals
"""

###############################################################################
# APPLY FILTER
low_freq = 0.1
"""
The low-frequency cut-off in the band-pass filtering step.
"""


high_freq = 30.0
"""
The high-frequency cut-off in the band-pass filtering step.
"""

###############################################################################
# PREPARE RAW

bipolar_channels = {'HEOG': ('HEOG_left', 'HEOG_right'),
                    'VEOG': ('VEOG_lower', 'FP2')}
"""
A dictionary of channels that defines how to combine two EOG channels into a single, bipolar EOG channel.
"""               

drop_channels = ['HEOG_left', 'HEOG_right', 'VEOG_lower']
"""
A list of channels to be removed from raw.
Here: After computation of bipolar references, only EOG channels should be removed, but not channel FP2.
"""

eog_channels = ['HEOG', 'VEOG']
"""
The name of the EOG channels as they result from the computation of the bipolar references using config-Parameter bipolar_channels.
"""

reference = ['P9', 'P10']
"""
The reference for EEG data: (str | List(str) | 'average')
A custom EEG reference is only used for ERP peak analysis. 
'Average' EEG reference is automatically used for the tasks decode and source_localization.
"""

###############################################################################
# RUN ICA

ica_reject = dict(eeg=350e-6, eog=500e-6)
"""
The reject parameter to exclude data from ICA fitting based on the peak-to-peak amplitude in the continuous data.
"""

ica_max_iterations = 1000
"""
The maximum number of iterations during decomposition of data into independent components.
"""

ica_method = 'infomax'
"""
The ICA method to use in the fit method: (fastica | infomax)
"""

ica_random_state = 5
"""
The seed for the NumPy random number generator. 
A value is explicitly passed to be able to compare the outputs of different runs.
"""

ica_eog_threshold = 2.0
""""
The threshold  above which a feature is classified as outlier. (Default: 3.0)
for use during detection mof EOG related components.
The lower the threshold value is set, the more ICs are identified as EGO-related.
"""

###############################################################################
# APPLY ICA

###############################################################################
# MAKE EPOCHS

event_of_interest = 'stimulus'
"""
An identifier of the events whose epochs are to be processed further as listed in the file events.tsv.
"""

incorrect_response_key = 'response:202'
"""
The identifier of the events that mark an incorrect behavioural response of the subject.
"""

response_window = (0.1, 1.0)
"""
The acceptable response window after the onset of the stimulus.
If a response is not received within the specified time window, the associated stimulus event 
is excluded from Epochs generation.
"""

epochs_tmin = -0.2 #-0.1
"""
The beginning of the epochs in seconds, relative to the time-locked event.
"""
epochs_tmax = 0.8
"""
The end of the epochs in seconds, relative to the time-locked event.
"""

reject_by_annotation = True
"""
If True, epochs overlapping with segments whose description begins with 'bad' are rejected.
If False, no rejection based on annotations is performed.
"""

reject_by_criteria = dict(eeg=400e-6, eog=500e-6)
"""
If set, epochs are rejected based on maximum peak-to-peak signal amplitude, 
i.e. the absolute difference between the lowest and the highest signal value. 
e.g. dict(eeg=300e-6, eog=150e-6)
"""

conditions = ['Distractor', 'Target']
"""
The list of the two conditions to be analyzed.
"""

distractors = ['stimulus:21', 'stimulus:31', 'stimulus:41', 'stimulus:51', 'stimulus:12', 'stimulus:32', 'stimulus:42', 'stimulus:52', 'stimulus:13', 'stimulus:23',
               'stimulus:43', 'stimulus:53', 'stimulus:14', 'stimulus:24', 'stimulus:34', 'stimulus:54', 'stimulus:15', 'stimulus:25', 'stimulus:35', 'stimulus:45']
"""
The keys of the events associated with the first condition.
"""

targets = ['stimulus:11', 'stimulus:22', 'stimulus:33', 'stimulus:44', 'stimulus:55']
"""
The keys of the events associated with the second condition.
"""

###############################################################################
# MAKE EVOKEDS
baseline = (None, 0)
"""
The time interval to use for baseline correction of epochs.
"""

bsl_regression = True
"""
If True, perform baseline regression on the channel of interest. Plots are provided to compare the traditional approach with the regression-based approach. 
If False, only perform traditional baselining.
In both cases, the subsequent tasks always consume epochs in which a traditional baseline correction was performed.
"""

###############################################################################
# ANALYZE SINGLE SUBJECT

meas_tmin = 0.3
"""
The a-priori assumed start time to measure the ERP component to be analyzed.
"""

meas_tmax = 0.6
"""
The a-priori assumed end time to measure the ERP component to be analyzed.
"""

test_tmin = 0.0
"""
The a-priori assumed start time to peform statistical tests.
"""

test_tmax = 0.6
"""
The a-priori assumed end time to peform statistical tests.
"""

n_permutations_t = 50000
"""
The number of permutations that are tested during the t-test.
"""

n_permutations_cluster = 1000

"""
The number of permutations that are tested during the permutation cluster test.
"""

thresh_cluster = 6.0
"""
The value to be used as the cluster forming threshold. The pipeline requires a numeric cluster forming threshold, which means that vertices with data values more extreme than threshold will be used to form clusters.
"""

###############################################################################
# RUN DECODING

resample  = 200 
"""
The sampling rate to be used for resampling the data in LDA and Logistic Regression approaches.
"""

resample_temp_gen = 50 
"""
The sampling rate to be used for resampling the data in Temporal Generalization.
"""

scoring = 'roc_auc'
"""
The evaluation function used to estimate the classification performance for all implemented approaches.
(roc_auc | accuracy | average_precision | recall)
"""

n_jobs = 4 
"""
The number of jobs to run in parallel.
"""

cv_fold = 5 
"""
The number of folds of the cross-validation splitting strategy.
"""

reg_csp = 0.2
"""
The value used for regularization for covariance estimation.
"""

###############################################################################
# SOURCE LOCALIZATION

snr_sur = 10.0
method_sur = 'dSPM'

snr_vol = 3.0
method_vol = 'dSPM'

create_video_for_subject = '002'
"""
The list of subjects for which videos will be created.
"""


###############################################################################
# COLORS

#palette = sns.color_palette() 
palette = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), (0.5803921568627451, 0.403921568627451, 0.7411764705882353), (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]
color_dict = dict(Distractor=palette[1],Target=palette[0],Difference=palette[4])
color_dict_td = dict(Distractor=color_dict['Distractor'], Target=color_dict['Target'])

###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# Some directories
fname.add('raw_data_dir', raw_data_dir)
fname.add('provided_data_dir', './provided')
fname.add('processed_data_dir', './processed')

# The data files that are used by the analysis steps
fname.add('events', '{raw_data_dir}/{task}/sub-{subject}/ses-{task}/eeg/sub-{subject}_ses-{task}_task-{task}_events.tsv')
fname.add('event_code_values', '{raw_data_dir}/{task}/task-{task}_events.json')
fname.add('badComponents_precomputed', '{raw_data_dir}/{task}/sub-{subject}/ses-{task}/eeg/sub-{subject}_ses-{task}_task-{task}_ica.tsv')
fname.add('ica_result_precomputed', '{raw_data_dir}/{task}/sub-{subject}/ses-{task}/eeg/sub-{subject}_ses-{task}_task-{task}_ica.set')
fname.add('badChannels_precomputed', '{raw_data_dir}/{task}/sub-{subject}/ses-{task}/eeg/sub-{subject}_ses-{task}_task-{task}_badChannels.tsv')
fname.add('badSegments_precomputed', '{raw_data_dir}/{task}/sub-{subject}/ses-{task}/eeg/sub-{subject}_ses-{task}_task-{task}_badSegments.csv')

# The data files resulting from manual annotation and labelling
fname.add('badChannels_manual', '{provided_data_dir}/{subject}-manual-badChannels.tsv')
fname.add('badSegments_manual', '{provided_data_dir}/{subject}-manual-badSegments.csv')
fname.add('badSegments_manual_txt', '{provided_data_dir}/{subject}-manual-badSegments.txt')
fname.add('badComponents_manual', '{provided_data_dir}/{subject}-manual-badComponents.tsv')
fname.add('volume_source_space', '{provided_data_dir}/volume_source_space.fif')

# The data files that are saved by steps of the pipeline
fname.add('filtered', '{processed_data_dir}/{subject}-filtered-raw.fif')
fname.add('prepared', '{processed_data_dir}/{subject}-prepared-raw.fif')
fname.add('ica_result_manual', '{processed_data_dir}/{subject}-ica.fif')
fname.add('badComponents_eog', '{processed_data_dir}/{subject}-eog-badComponents.tsv')
fname.add('raw_cleaned', '{processed_data_dir}/{subject}-cleaned-raw.fif')
fname.add('epoched', '{processed_data_dir}/{subject}-epo.fif')
fname.add('evokeds', '{processed_data_dir}/{subject}-ave.fif')
fname.add('grand_ave', '{processed_data_dir}/grand-ave.fif')
fname.add('decoded', '{processed_data_dir}/{subject}-decoding.fif')

# Source localization data
fname.add('noise_covariance', '{processed_data_dir}/{subject}-noise-covariance.fif')
fname.add('data_covariance', '{processed_data_dir}/{subject}-data-covariance.fif')
fname.add('forward_solution_surface', '{processed_data_dir}/{subject}-forward-solution-surface.fif')
fname.add('forward_solution_volume', '{processed_data_dir}/{subject}-forward-solution-volume.fif')
fname.add('source_estimate_evoked_total_surface', '{processed_data_dir}/{subject}-source-estimate-evoked-total-surface')
fname.add('source_estimate_evoked_total_volume', '{processed_data_dir}/{subject}-source-estimate-evoked-total-volume')
fname.add('source_estimate_evoked_distractor_surface', '{processed_data_dir}/{subject}-source-estimate-evoked-distractor-surface')
fname.add('source_estimate_evoked_target_surface', '{processed_data_dir}/{subject}-source-estimate-evoked-target-surface')
fname.add('source_estimate_evoked_difference_surface', '{processed_data_dir}/{subject}-source-estimate-evoked-difference-surface')

# Filenames for MNE reports
fname.add('reports_dir', './reports')
fname.add('report_preprocessing', '{reports_dir}/{subject}-report-preprocessing.h5')
fname.add('report_manual_cleaning', '{reports_dir}/report-manual-cleaning.h5')
fname.add('report_analysis_single', '{reports_dir}/{subject}-report-analysis-single.h5')
fname.add('report_analysis_all', '{reports_dir}/report-analysis-all.h5')
fname.add('report_decoding', '{reports_dir}/{subject}-report-decoding.h5')
fname.add('report_source_localization', '{reports_dir}/{subject}-report-source-localization.h5')
fname.add('source_movie_evoked_difference', '{subject}-source-movie-evoked-difference.mp4')

fname.add('report_preprocessing_html', '{reports_dir}/{subject}-report-preprocessing.html')
fname.add('report_manual_cleaning_html', '{reports_dir}/report-manual-cleaning.html')
fname.add('report_analysis_single_html', '{reports_dir}/{subject}-report-analysis-single.html')
fname.add('report_analysis_all_html', '{reports_dir}/report-analysis-all.html')
fname.add('report_decoding_html', '{reports_dir}/{subject}-report-decoding.html')
fname.add('report_source_localization_html', '{reports_dir}/{subject}-report-source-localization.html')
fname.add('introduction_html', '{reports_dir}/introduction.html')
fname.add('references_html', '{reports_dir}/references.html')

# Source localization output
fname.add('source_localization_cheatsheet_html', '{reports_dir}/source-localization-cheatsheet.html')
fname.add('source_localization_movie_html', '{reports_dir}/{subject}-source-localization-movie.html')

# File produced by check_system.py
fname.add('system_check', './system_check.txt')
