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

epochs = mne.read_epochs(cfg.fname.epoched_cleaned(subject=subject), preload=True)

# Problem: When applying baseline correction?
# print(f'Epochs baseline: {epochs.baseline}')
# print(f'Evoked baseline: {epochs.average().baseline}')

# Create evoked data sets by averaging different conditions (Here: Targets and Distractors)
all_evoked = dict()
for e in epochs.event_id.keys():
    evoked_mean = epochs[e].average()
    all_evoked[e] = evoked_mean

evokeds_mean = list(all_evoked.values())
mne.write_evokeds(cfg.fname.evoked(subject=subject), evokeds_mean, overwrite=True)

# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report(subject=subject)) as report:
    # Missing: add evokeds information to report
    # HERE: e.g. for subset of evokeds
    # report.add_evokeds(evokeds=evokeds)
    for e in epochs.event_id.keys():
        fig1 = all_evoked[e].plot(ylim=dict(eeg=(-20, 20)), spatial_colors=True, show=False)
        report.add_figure(fig1,'Evoked ' + e, replace=True)
   
    # Compare ERPs of targets and distractors
    fig2 = mne.viz.plot_compare_evokeds({'Targets': all_evoked['Targets'],
                                        'Distractors': all_evoked['Distractors']},
                                        picks=cfg.channels_to_analyze,
                                        show=False)
    report.add_figure(fig2,'Comparison of ERPs of channels of interest '+ str(cfg.channels_to_analyze), replace=True)

    # Compare ERPs of targets and distractors
    fig3 = mne.viz.plot_compare_evokeds({'mean': all_evoked['Targets'],
                                        'median': epochs['Targets'].average(method=utils.median),
                                        'winsorized': epochs['Targets'].average(method=utils.winsor)},
                                        picks=cfg.channels_to_analyze,
                                        show=False)
    report.add_figure(fig3,'Comparison of different averaging methods for '+ str(cfg.channels_to_analyze), replace=True)

    report.save(cfg.fname.report_html(subject=subject), overwrite=True, open_browser=False)
