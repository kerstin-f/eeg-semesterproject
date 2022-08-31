"""
Run ICA on raw data filtered with 1 Hz highpass.
"""
import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level) 
import pandas as pd
import os
import utils

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Load prepared raw data of the subject to be processed manually
raw = mne.io.read_raw_fif(cfg.fname.prepared(subject=subject), preload=True)
# 
raw_filtered = raw.copy().filter(l_freq=1.0,h_freq=cfg.high_freq,fir_design='firwin', phase='zero')

ica_n_components = len(raw.ch_names) - len(cfg.eog_channels) - len(cfg.reference)
ica = mne.preprocessing.ICA(n_components=ica_n_components, method=cfg.ica_method, random_state=cfg.ica_random_state)
ica.fit(raw_filtered, reject_by_annotation=True, reject=cfg.ica_reject, verbose=True)

explained_variance = ica.pca_explained_variance_[:ica.n_components_].sum()/ica.pca_explained_variance_.sum()
print('>>> Actual number of PCA components used for ICA decomposition: {}'.format(ica.n_components_))
print('>>> Actual number of iterations required to complete ICA: {}'.format(ica.n_iter_))
print('>>> The PCA components used for ICA decomposition explain {}% of the variance.'.format(round(explained_variance*100, 2)))

# Save ICA solution
ica.save(cfg.fname.ica_result_manual(subject=subject), overwrite=True)

# Automatically identify EOG artifacts from ICA components
# Only executed if no file with manually identified bad components is available
if not os.path.isfile(cfg.fname.badComponents_manual(subject=subject)):
        eog_epochs= mne.preprocessing.create_eog_epochs(raw, ch_name=cfg.eog_channels, reject_by_annotation=True, reject=cfg.ica_reject)
        # Find the ICs that best match the EOG signal
        eog_idx, scores = ica.find_bads_eog(eog_epochs, threshold=cfg.ica_eog_threshold, ch_name=cfg.eog_channels)
        df_eog = pd.DataFrame(list(set(eog_idx)))
        # Save EOG-related bad components
        df_eog.to_csv(cfg.fname.badComponents_eog(subject=subject), sep="\t", index=False)
if cfg.generate_reports:
    # Add a plot of the data to the HTML report
    with mne.open_report(cfg.fname.report_preprocessing(subject=subject)) as report:
        utils.add_html(report, 'run_ica.implementation','Task run_ica', 'Manual_ICA')
        report.save(cfg.fname.report_preprocessing_html(subject=subject),overwrite=True,open_browser=False)