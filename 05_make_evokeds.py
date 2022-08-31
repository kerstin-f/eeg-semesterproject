"""
Create evoked datasets by averaging over different subsets of epochs.
"""
import utils
import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level)
from itertools import islice
import numpy as np
from matplotlib import pyplot as plt

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###',help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Load epochs data
epochs = mne.read_epochs(cfg.fname.epoched(subject=subject), preload=True)

assert (cfg.baseline[0] == None or cfg.baseline[0] >= cfg.epochs_tmin) and cfg.baseline[1] <= cfg.epochs_tmax, 'The baseline period must lie within the Epochs limits.'
assert isinstance(cfg.choi,str) and cfg.choi in epochs.ch_names, 'Channel to be plottet not in epochs.'

# Create evoked data sets by averaging per condition
evokeds_dict_traditional = dict.fromkeys(cfg.conditions)
for condition in cfg.conditions:
    _evoked = epochs[condition].average().apply_baseline(cfg.baseline)
    _evoked.comment = condition
    evokeds_dict_traditional[condition] = _evoked

# Compute difference wave by weighted subtraction of the averages over conditions
_evoked_diff = mne.combine_evoked(list(evokeds_dict_traditional.values())[:2], weights=[-1, 1])
_evoked_diff.comment = 'Difference'
evokeds_dict_traditional['Difference'] = _evoked_diff
# Compute total wave without considering the conditions 
_evoked_total = epochs.average().apply_baseline(cfg.baseline)
_evoked_total.comment = 'Total'
evokeds_dict_traditional['Total'] = _evoked_total

print('>>> Writing {} evoked datasets to disk ({}).'.format(len(evokeds_dict_traditional), ','.join(str(e) for e in cfg.conditions)+ ' Difference,Total'))
mne.write_evokeds(cfg.fname.evokeds(subject=subject),list(evokeds_dict_traditional.values()), overwrite=True)

# For comparison: compute regression-based baseline correction (over one channel)
if cfg.bsl_regression:
    evokeds_dict_bsl_regression = dict.fromkeys(cfg.conditions)
    regression_model = utils.compute_regression_baselining(epochs, cfg.choi)
    for condition in cfg.conditions:
        evokeds_dict_bsl_regression[condition] = regression_model[condition].beta

if cfg.generate_reports:
    # Add a plot of the data to the HTML report
    with mne.open_report(cfg.fname.report_preprocessing(subject=subject)) as report:       
        # ---------------------------------------------------------------------------------------------------------
        fig_evoked = evokeds_dict_traditional['Total'].plot(ylim=dict(eeg=(-20, 20)), spatial_colors=True, show=False)
        
        # ---------------------------------------------------------------------------------------------------------
        times = np.arange(0, cfg.epochs_tmax, 0.1)
        fig_topomaps = [evokeds_dict_traditional[condition].plot_topomap(times=times, average=0.050, vmin=-20, vmax=20, show=False) for condition in cfg.conditions]

        # ---------------------------------------------------------------------------------------------------------
        kwargs = dict(picks=cfg.choi, truncate_yaxis=False, legend='lower right', show_sensors='upper right', show=False)
        #color_dict = {'Distractor': 'blue', 'Target':'red'}
        linestyle_dict = {'Distractor':'--', 'Target':'-'}
        fig_comparison = mne.viz.plot_compare_evokeds([evokeds_dict_traditional.get(condition) for condition in cfg.conditions], colors=cfg.color_dict_td, linestyles=linestyle_dict, **kwargs)

        fig_diff = mne.viz.plot_compare_evokeds(evokeds_dict_traditional['Difference'], colors=[cfg.color_dict['Difference']], **kwargs)
        
        # ---------------------------------------------------------------------------------------------------------
        if cfg.bsl_regression:
            fig_comparison_regression = mne.viz.plot_compare_evokeds(evokeds_dict_bsl_regression, colors=cfg.color_dict_td, linestyles=linestyle_dict, **kwargs)
            baseline_effect= regression_model['Baseline'].beta
            fig_baseline_effect = baseline_effect.plot(picks=cfg.choi, 
                                                        hline=[1.], 
                                                        units=dict(eeg=r'$\beta$ value'),
                                                        titles=dict(eeg=cfg.choi), 
                                                        selectable=False,
                                                        show=False)
        
        # ---------------------------------------------------------------------------------------------------------
        
        utils.add_html(report, 'make_evokeds.implementation','Task make_evokeds', 'Evokeds')
        report.add_figure(fig_evoked, 'Evoked (Total)', replace=True, tags='Evokeds')
        report.add_figure(fig_topomaps, title='Scalp topography maps', caption=cfg.conditions, replace=True, tags='Evokeds')
        title_left = 'Evokeds ({})'.format(cfg.choi)
        title_right = 'Difference wave ({})'.format(cfg.choi)
        fig_combined_erps = utils.combine_figures(fig_comparison, title_left, fig_diff, title_right)
        report.add_figure(fig_combined_erps, 'Comparison of Evokeds')
        if cfg.bsl_regression:
            title_left = 'Evokeds after regression-based baseline correction ({})'.format(cfg.choi)
            title_right = 'Estimated effect of the baseline period ({})'.format(cfg.choi)
            fig_combined_baseline = utils.combine_figures(fig_comparison_regression, title_left, fig_baseline_effect , title_right)
            report.add_figure(fig_combined_baseline, 'Baseline effect', replace=True, tags='Evokeds')
        report.save(cfg.fname.report_preprocessing_html(subject=subject), overwrite=True, open_browser=False)
