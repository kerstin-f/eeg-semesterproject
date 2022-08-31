"""
Create epochs by segementing data according to event codes.
"""
import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level) 
import numpy as np
import pandas as pd
import utils
from matplotlib import pyplot as plt

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Load cleand raw data
raw = mne.io.read_raw_fif(cfg.fname.raw_cleaned(subject=subject), preload=True)

assert cfg.response_window[0]>0 and cfg.response_window[0]<cfg.response_window[1], 'Acceptable response time window does not make sense.'
assert cfg.epochs_tmin < cfg.epochs_tmax, 'Start and end time of epochs does not make sense.'
assert isinstance(cfg.choi, str) and cfg.choi in raw.ch_names, 'Channel to be plottet not in raw.'
assert isinstance(cfg.conditions,list) and len(cfg.conditions)==2, 'The pipeline is designed to process two conditions provided as list of strings.'
assert isinstance(cfg.distractors, list) and isinstance(cfg.targets, list), 'The event codes of the two conditions must be provided as a list of strings.'

# Focus on activity related to stimulus events
evts, evts_dict = mne.events_from_annotations(raw)
stim_keys = [evt for evt in evts_dict.keys() if cfg.event_of_interest in evt]
evts_dict_stim = dict((key, evts_dict[key]) for key in stim_keys if key in evts_dict)

# Compute indices of responses that lie outside the acceptable time window
df_events = pd.read_csv(cfg.fname.events(subject=subject, task=cfg.task), sep='\t')
df_events['Diff'] = df_events.onset.diff()
df_exclude = pd.DataFrame()
df_exclude = df_events.iloc[df_events.query('Diff<{} or Diff>{}'.format(cfg.response_window[0],cfg.response_window[1])).index.tolist()]
ix_response = df_exclude.query("trial_type=='response'").index.tolist()

drop_incorrect=0
drop_dubious=0
evts_cleaned = evts
for ix,evt in enumerate(evts):
    # Exclude stimulus events followed by a response that lies outside the acceptable time window
    if ix in ix_response:
        evts_cleaned = np.delete(evts,ix-1,0)
        drop_dubious += 1
    # Exclude stimulus events followed by an incorrect behavioral answer
    if evts_dict[cfg.incorrect_response_key] == evt[2]:
        evts_cleaned = np.delete(evts,ix-1,0)
        drop_incorrect += 1

print('>>> {} event(s) dropped because response is outside acceptable response window of {} [s].'.format(drop_dubious, cfg.response_window))
print('>>> {} event(s) dropped due to false behavioral response.'.format(drop_incorrect))

# Epoch the data using different rejection settings
# epochs_dict['annotation']: reject epochs based on annotated bad time segments
# epochs_dict['threshold: 300e-6']: reject epochs based on peak-to-peak rejection threshold
# epochs_dict['config']: reject epochs based on specifications in config
by_annotation = [True, False, cfg.reject_by_annotation]
by_threshold = [None, dict(eeg=100e-6), cfg.reject_by_criteria]
epochs_dict = dict.fromkeys(['annotation', 'threshold: 100e-6', 'config'])

for ix, key in enumerate(epochs_dict):
    _epochs = mne.Epochs(raw, 
                        events=evts_cleaned, 
                        event_id=evts_dict_stim,
                        tmin=cfg.epochs_tmin, 
                        tmax=cfg.epochs_tmax,
                        proj=True,
                        baseline=None,
                        reject=by_threshold[ix],
                        reject_by_annotation=by_annotation[ix])
    epochs_dict[key] = _epochs
epochs_dict['config'].load_data()

# Rename event labels and event codes to facilitate further programming
epochs_combined = mne.epochs.combine_event_ids(epochs_dict['config'], cfg.distractors, {cfg.conditions[0]: 0}, copy=True)
epochs_combined = mne.epochs.combine_event_ids(epochs_combined, cfg.targets, {cfg.conditions[1]: 1}, copy=True)
epochs_combined = epochs_combined[[cfg.conditions[0],cfg.conditions[1]]]

print('>>> Writing {} epochs to disk.'.format(len(epochs_combined)))
epochs_combined.save(cfg.fname.epoched(subject=subject), overwrite=True)

if cfg.generate_reports:
    # Add a plot of the data to the HTML report
    with mne.open_report(cfg.fname.report_preprocessing(subject=subject)) as report:
        mne.viz.set_browser_backend('matplotlib') # qt

        # ---------------------------------------------------------------------------------------------------------
        fig_events = mne.viz.plot_events(evts,
                                         sfreq=raw.info['sfreq'],
                                         first_samp=raw.first_samp,
                                         event_id=evts_dict,
                                         show=False, verbose=cfg.mne_log_level)
        fig_events.set_size_inches(12, 8)

        # ---------------------------------------------------------------------------------------------------------
        # Plot 10 epochs
        num_cols = 30
        fig_epochs = epochs_dict['config'].plot(n_epochs=10, show_scrollbars=False, show_scalebars=False, show=False, overview_mode='hidden')
        fig_epochs.set_size_inches(12, 8)

        # ---------------------------------------------------------------------------------------------------------
        # Compare total evokeds derived from different rejections methods
        evokeds_dict = {key:_epochs.average() for key, _epochs in epochs_dict.items()}
        kwargs = dict(picks=cfg.choi, truncate_yaxis=False, legend='lower right', show_sensors='upper right', show=False)
        fig_comparison = mne.viz.plot_compare_evokeds(evokeds_dict, **kwargs)

        # ---------------------------------------------------------------------------------------------------------
        utils.add_html(report, 'make_epochs.implementation','Task make_epochs', 'Epochs')
        report.add_epochs(epochs=epochs_combined, title='Epochs info', psd=False, replace=True, tags='Epochs')
        report.add_figure(fig_events,'Events', replace=True, tags='Epochs')
        report.add_figure(fig_epochs,'10 Epochs', replace=True, tags='Epochs')
        report.add_figure(fig_comparison,'Comparison of Evokeds ({})'.format(cfg.choi), replace=True, tags='Epochs')
        report.save(cfg.fname.report_preprocessing_html(subject=subject),overwrite=True,open_browser=False)
