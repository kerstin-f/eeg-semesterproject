"""
Perform bandpass filtering.
"""
import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level) 
import utils

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Load filtered raw data and create epochs
raw = mne.io.read_raw_fif(cfg.fname.filtered(subject=subject), preload=True)

# Account for precomputed bad data
# https://mne.tools/dev/auto_tutorials/preprocessing/15_handling_bad_channels.html
badAnnotations, badChannels = utils.load_precomputed_badData(subject)
raw.annotations.append(badAnnotations.onset,badAnnotations.duration,badAnnotations.description)
raw.info['bads'].extend(badChannels)
raw.interpolate_bads(reset_bads=False)

# Set an EEG reference
raw.set_eeg_reference('average', projection=True)

raw.save(cfg.fname.prepared(subject=subject), overwrite=True)

print("HERE")
