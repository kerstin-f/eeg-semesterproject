"""
Remove effects of ICs from raw that are marked for exclusion.
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

# Load prepared raw data of the subject to be processed
raw = mne.io.read_raw_fif(cfg.fname.prepared(subject=subject), preload=True)

# Distinguish between subjects to be cleaned manually or via precomputed files
manual = subject in utils.generate_subjects_list(cfg.subjects_manual)

# Load ICA decomposition and bad components of the subject
ica = utils.load_ica(subject, manual)
if not manual: 
    ica = utils.add_ica_info(raw, ica)
badComponents = utils.load_badComponents(subject, manual)
print('>>> The following artifactual ICs will be excluded: {}'.format(badComponents))
ica.exclude = badComponents

# Reconstruct raw with bad components exluded
cleaned_raw = ica.apply(raw.copy())

# Save reconstructed raw
cleaned_raw.save(cfg.fname.raw_cleaned(subject=subject), overwrite=True)

if cfg.generate_reports:
    # Add a plot of the data to the HTML report
    with mne.open_report(cfg.fname.report_preprocessing(subject=subject)) as report:
        # ---------------------------------------------------------------------------------------------------------     
        fig_components = ica.plot_components(show=False)
        
        # ---------------------------------------------------------------------------------------------------------
        fig_sources = ica.plot_sources(raw.copy().filter(l_freq=1., h_freq=None), start=100, stop=120, show_scrollbars=False, show=False)
        # ---------------------------------------------------------------------------------------------------------
        utils.add_html(report, 'run_ica.implementation','Task run_ica', 'Manual_ICA')
        utils.add_html(report, 'apply_ica.implementation','Task apply_ica', 'ICA')
        report.add_ica(ica, 'ICA info', inst=raw, picks=ica.exclude, tags='ICA')
        report.add_figure(fig_components,'ICA components', tags='ICA')
        report.add_figure(fig_sources, 'ICA sources', tags='ICA')
        report.save(cfg.fname.report_preprocessing_html(subject=subject), overwrite=True, open_browser=False)