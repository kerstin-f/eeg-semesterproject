"""
Perform bandpass filtering.
"""
import argparse
import mne
import numpy as np
from matplotlib import pyplot as plt
import config as cfg
import utils
from autoreject import AutoReject

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Extract epochs for one subject.
# Loading filtered raw data and creating epochs.
# Only events corresponding to `conditions` will be used to create epochs.
raw = mne.io.read_raw_fif(cfg.fname.filtering(subject=subject), preload=True)

# evts: sample, duration, event_id
evts, evts_dict = mne.events_from_annotations(raw)

stim_keys = [e for e in evts_dict.keys() if "stimulus" in e]
evts_dict_stim = dict((k, evts_dict[k]) for k in stim_keys if k in evts_dict)


# Epoch the data
# (HERE: without rejection (manual | thresh))

# epochs:           no rejection
# epochs_manual:    rejection by manual annotation
# epochs_thresh:    peak-to-peak rejection
# epochs_ar:        autoreject

# Get epochs without manual rejection
epochs = mne.Epochs(raw, 
                    events=evts, 
                    event_id=evts_dict_stim,
                    tmin=cfg.epochs_tmin, 
                    tmax=cfg.epochs_tmax,
                    reject_by_annotation=False)

# Get epochs with manual rejection
epochs_manual = mne.Epochs(raw, 
                    events=evts, 
                    event_id=evts_dict_stim,
                    tmin=cfg.epochs_tmin, 
                    tmax=cfg.epochs_tmax,
                    reject_by_annotation=True)

# Get epochs with rejection based on rejection criterion for a peak-to-peak rejection method
reject_criteria = dict(eeg=200e-6)       # 100 µV # HAD TO INCREASE IT HERE, 100 was too harsh
                                         # 200 µV
epochs_tresh = mne.Epochs(raw, 
                    events=evts, 
                    event_id=evts_dict_stim,
                    tmin=cfg.epochs_tmin, 
                    tmax=cfg.epochs_tmax,
                    reject=reject_criteria,
                    reject_by_annotation=False)

# Get epochs with autorejection
# ar = AutoReject(verbose='tqdm')
# epochs.load_data()
# epochs_ar = ar.fit_transform(epochs)
epochs_ar = epochs

# Remove reference to raw and free memory
epochs.load_data()

# Missing: metadata query

# Set an EEG reference
projection = True if cfg.eeg_reference == 'average' else False
epochs.set_eeg_reference(cfg.eeg_reference, projection=projection)

print(f'Writing {len(epochs)} epochs to disk.')
epochs.save(cfg.fname.epoching(subject=subject), overwrite=True)

# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report(subject=subject)) as report:
    # Missing: Plot raw data and power spectral density.
    report.add_epochs(epochs=epochs, title='Extracted epochs', psd=False, replace=True)

    fig1 = mne.viz.plot_events(evts, 
                              sfreq=raw.info['sfreq'],
                              first_samp=raw.first_samp, 
                              event_id=evts_dict, show=False)
    report.add_figure(fig1,'Events', replace=True)
    
    # Plot all trials 'manually', without using mne's functionality
    fig2 = plt.figure()

    # Get all epochs as a 3D array (n_epochs, n_channels, n_times)
    plt.plot(np.squeeze(epochs.get_data(picks=cfg.plot_channel_epoching, units='uV')[0:5,0,1:100].T))
    plt.xlabel("Time")
    plt.ylabel("µV")
    report.add_figure(fig2,'Epochs 0-5 of channel '+ cfg.plot_channel_epoching, replace=True)

    # Compare epochs_thresh with  manual rejection and with the ERPs without rejection
    fig2 = mne.viz.plot_compare_evokeds({'raw':epochs.average(),
                                         'clean':epochs_manual.average(),
                                         'thresh':epochs_tresh.average(),
                                         'autoreject': epochs_ar.average(),
                                         'robust':epochs.load_data().average(method=utils.winsor),
                                         'median':epochs.load_data().average(method=utils.median)},
                                         picks=cfg.plot_channel_epoching,
                                         show=False)
    report.add_figure(fig2,'Comparison of ERPs '+ cfg.plot_channel_epoching, replace=True)

    report.save(cfg.fname.report_html(subject=subject),overwrite=True,open_browser=False)
