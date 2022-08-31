"""
Prepare raw (including annotation of bad segments and bad channels, re-referencing and setting of channel locations).
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

# Load filtered raw data of the subject to be processed
raw = mne.io.read_raw_fif(cfg.fname.filtered(subject=subject), preload=True)

assert isinstance(cfg.eog_channels, list), 'EOG channels must be provided as a list of strings.'
assert isinstance(cfg.choi, str) and cfg.choi in raw.ch_names, 'Channel to be plottet not in raw.'

# Account for bad time segments and bad data
manual = subject in utils.generate_subjects_list(cfg.subjects_manual)
badSegments = utils.load_badSegments(subject, manual)
raw.annotations.append(badSegments.onset,badSegments.duration,badSegments.description)
badChannels_ix = utils.load_badChannels(subject, manual)
badChannels = [raw.ch_names[ix] for ix in badChannels_ix]
print('>>> The following channels are marked as bad: {}'.format(badChannels))
raw.info['bads'].extend(badChannels)

# Create bipolar channels for use as EOG channels
for ch, (anode, cathode) in cfg.bipolar_channels.items():
    mne.set_bipolar_reference(raw, anode=anode, cathode=cathode, ch_name=ch, drop_refs=False,copy=False)
raw.drop_channels(cfg.drop_channels)
# Set channel type of new bipolar channels 
raw.set_channel_types({name:'eog' for name in cfg.eog_channels})

# Add channel locations
raw.set_montage('standard_1020',match_case=False)

# Perform re-referencing
raw.set_eeg_reference(cfg.reference)

# Save prepared raw
raw.save(cfg.fname.prepared(subject=subject), overwrite=True)

# Add a plot of the data to the HTML report
if cfg.generate_reports:
    with mne.open_report(cfg.fname.report_preprocessing(subject=subject)) as report:
        # ---------------------------------------------------------------------------------------------------------
        mne.viz.set_browser_backend('matplotlib') # qt

        fig_raw = raw.plot(show_scrollbars=False)

        # ---------------------------------------------------------------------------------------------------------
        # Form the 10-20 montage
        montage_1020 = mne.channels.make_standard_montage('standard_1020')
        # Keep only channels that are present in given dataset 
        ind = [i for (i, channel) in enumerate(montage_1020.ch_names) if channel in raw.ch_names]
        montage_1020_cust = montage_1020.copy()
        montage_1020_cust.ch_names = [montage_1020.ch_names[x] for x in ind]
        kept_channel_info = [montage_1020.dig[x+3] for x in ind]
        montage_1020_cust.dig = montage_1020.dig[0:3]+kept_channel_info
        fig_montage = mne.viz.plot_montage(montage_1020_cust, show=False)

        # ---------------------------------------------------------------------------------------------------------
        utils.add_html(report, 'prepare_raw.implementation','Task prepare_raw', 'Filtering')
        report.add_raw(raw=raw, title='Raw info', psd=False, butterfly=False, replace=True, tags='Preparation')
        report.add_figure(fig_montage,'Montage 10-20', replace=True, tags='Preparation')
        report.add_figure(fig_raw, 'Raw', replace=True, tags='Preparation')
        report.save(cfg.fname.report_preprocessing_html(subject=subject),overwrite=True,open_browser=False)