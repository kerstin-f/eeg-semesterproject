"""
Perform bandpass filtering.
"""
import argparse
import mne
import config as cfg

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Load filtered raw data and create epochs
raw = mne.io.read_raw_fif(cfg.fname.prepared(subject=subject), preload=True)

# evts: sample, duration, event_id
evts, evts_dict = mne.events_from_annotations(raw)

# Focus on activity related to stimulus events 
stim_keys = [e for e in evts_dict.keys() if "stimulus" in e]
evts_dict_stim = dict((k, evts_dict[k]) for k in stim_keys if k in evts_dict)

# Epoch the data
# Get epochs without rejection
epochs = mne.Epochs(raw, 
                    events=evts, 
                    event_id=evts_dict_stim,
                    tmin=cfg.epochs_tmin, 
                    tmax=cfg.epochs_tmax,
                    baseline=None,
                    reject_by_annotation=False, 
                    preload=True)

# Get epochs with manual rejection
epochs_manual = mne.Epochs(raw, 
                    events=evts, 
                    event_id=evts_dict_stim,
                    tmin=cfg.epochs_tmin, 
                    tmax=cfg.epochs_tmax,
                    baseline=None,
                    reject_by_annotation=True,
                    preload=True)
                    
# Get epochs with rejection based on rejection criterion for a peak-to-peak rejection method
reject_criteria = dict(eeg=200e-6)       # 100 µV # HAD TO INCREASE IT HERE, 100 was too harsh
epochs_tresh = mne.Epochs(raw, 
                    events=evts, 
                    event_id=evts_dict_stim,
                    tmin=cfg.epochs_tmin, 
                    tmax=cfg.epochs_tmax,
                    baseline=None,
                    reject=reject_criteria,
                    reject_by_annotation=False,
                    preload=True)


epochs_combined = mne.epochs.combine_event_ids(epochs_manual, cfg.targets, {'Targets': 0}, copy=True)
epochs_combined = mne.epochs.combine_event_ids(epochs_combined, cfg.distractors, {'Distractors': 1}, copy=True)
epochs_combined = epochs_combined[['Targets','Distractors']]

print(f'Writing {len(epochs)} epochs to disk.')
epochs_combined.save(cfg.fname.epoched(subject=subject), overwrite=True)

# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report(subject=subject)) as report:
    report.add_epochs(epochs=epochs, title='Epoched data', psd=False, replace=True)

    fig1 = mne.viz.plot_events(evts, 
                              sfreq=raw.info['sfreq'],
                              first_samp=raw.first_samp, 
                              event_id=evts_dict, show=False)
    report.add_figure(fig1,'Events', replace=True)
    
    # Plot all trials 'manually', without using mne's functionality
    fig2 = epochs_manual.plot(n_epochs=10, show_scrollbars=False)
    report.add_figure(fig2,'10 Epochs (rejection based on annotation)', replace=True)

    # Compare epochs_thresh with  manual rejection and with the ERPs without rejection
    fig2 = mne.viz.plot_compare_evokeds({'without rejection':epochs.average(),
                                         'rejection based on annotation':epochs_manual.average(),
                                         'rejection based on thresh':epochs_tresh.average()},
                                         picks=cfg.plot_channel_epoching,
                                         show=False)
    report.add_figure(fig2,'Comparison of ERPs '+ cfg.plot_channel_epoching, replace=True)

    report.save(cfg.fname.report_html(subject=subject),overwrite=True,open_browser=False)
