"""
Create useful plots 
1. to manually analysze the ICA decomposition 
2. to check whether the artefactual components have been correctly identified
(Not part of the pipeline.)
"""

import config as cfg
import mne
mne.set_log_level(cfg.mne_log_level) 
import utils
import csv

subjects = ['002', '003', '004']
raw = []
ica = []

# Components identified as artifactual through manual inspection
badComponents = [[0,11], [0,2,21], [0,8,10,18]]
for ix, subject in enumerate(subjects):
    # Load prepared raw data
    raw.append(mne.io.read_raw_fif(cfg.fname.prepared(subject=subject), preload=True))
    ica.append(utils.load_ica(subject, True))
    print('>>> Manually labeled bad components:{} (of subject {})'.format(badComponents[ix], subject))

# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report_manual_cleaning) as report:
    report.title = 'Manual Cleaning of subjects 002, 003, 004'
    utils.add_html(report, 'manual_cleaning.annotate_raw','Annotation of bad time segments and channels', 'annotate_raw')
    utils.add_html(report, 'manual_cleaning.label_badComponents','Identification of artefactual components', 'label_badComponents')
    for ix, subject in enumerate(subjects):
        print('>>> Plot ICA for subject ' + subject)
        ica[ix].exclude = badComponents[ix]
        fig_sources = ica[ix].plot_sources(raw[ix].copy().filter(l_freq=1., h_freq=None), start=100, stop=120, show_scrollbars=False, show=False)
        fig_components = ica[ix].plot_components(show=False)
        if subject == '002':
            utils.add_html(report, 'manual_cleaning.workflow','Workflow for subject 2', subject)
        report.add_ica(ica[ix], 'ICA', inst=raw[ix], tags=subject)
        report.add_figure(fig_sources, 'ICA sources', tags=subject)
        report.add_figure(fig_components,'ICA components ({})'.format(subject), tags=subject)
    report.save(cfg.fname.report_manual_cleaning_html, overwrite=True, open_browser=False)

# Save identified bad components to tsv file
for ix, subject in enumerate(subjects):
    row = [[component] for component in badComponents[ix]]
    with open(cfg.fname.badComponents_manual(subject=subject), 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerows(row)
