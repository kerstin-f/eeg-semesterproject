
import config as cfg
import mne
mne.set_log_level(cfg.mne_log_level)
import utils

report = mne.Report(title='Introduction')
utils.add_html(report, 'introduction.task', 'Task', 'Task')
utils.add_html(report, 'introduction.implementation', 'Implementation', 'Implementation')
report.save(cfg.fname.introduction_html, overwrite=True, open_browser=False)

report = mne.Report(title='References')
utils.add_html(report, 'references.content', 'References', 'References')
report.save(cfg.fname.references_html, overwrite=True, open_browser=False)
