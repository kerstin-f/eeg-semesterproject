"""
Perform bandpass filtering.
"""
import argparse
import mne
import utils
import numpy as np
from mne_bids import (BIDSPath, read_raw_bids)
from matplotlib import pyplot as plt
from config import fname, conditions, epochs_metadata_tmin, epochs_metadata_tmax, eeg_reference

# All parameters are defined in config.py
from config import fname, task, bids_root, l_freq, h_freq, h_trans_bandwidth, l_trans_bandwidth

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Extract epochs for one subject.
# Loading filtered raw data and creating epochs.
# Only events corresponding to `conditions` will be used to create epochs.
raw = mne.io.read_raw_fif(fname.filtering(subject=subject), preload=True)

# evts: sample, duration, event_id
evts, evts_dict = mne.events_from_annotations(raw)

stim_keys = [e for e in evts_dict.keys() if "stimulus" in e]
evts_dict_stim = dict((k, evts_dict[k]) for k in stim_keys if k in evts_dict)

# Epoch the data
# Do not reject based on peak-to-peak or flatness thresholds at this stage
# (HERE: without rejection (manual | thresh))
epochs = mne.Epochs(raw, 
                    events=evts, 
                    event_id=evts_dict_stim,
                    tmin=epochs_metadata_tmin, 
                    tmax=epochs_metadata_tmax,
                    proj=False, 
                    baseline=None,
                    preload=False,
                    reject=None, 
                    reject_by_annotation=False)

# Remove reference to raw and free memory
epochs.load_data()
del raw

# Missing: metadata query

# Set an EEG reference
projection = True if eeg_reference == 'average' else False
epochs.set_eeg_reference(eeg_reference, projection=projection)

print(f'Writing {len(epochs)} epochs to disk.')
epochs.save(fname.epoching(subject=subject), overwrite=True)


# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    # Missing: Plot raw data and power spectral density.
    fig = mne.viz.plot_evoked(evoked=epochs.average(), show=False, picks="Cz")
    report.add_figure(fig,'Epoched data: Channel Cz')
    report.save(fname.report_html(subject=subject), overwrite=True,
                open_browser=False)