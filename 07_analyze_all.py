"""
Perform group analysis of ERP peak.
"""

import config as cfg
import mne
mne.set_log_level(cfg.mne_log_level)
import numpy as np
import utils
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


subjects = utils.generate_subjects_list(cfg.subjects)

evokeds_dict = {key: [] for key in cfg.conditions + ['Difference'] + ['Total']}
grand_dict = dict.fromkeys(evokeds_dict.keys())

for subject in subjects:
    evokeds_dict[cfg.conditions[0]].append(mne.read_evokeds(cfg.fname.evokeds(subject=subject), condition=cfg.conditions[0]))
    evokeds_dict[cfg.conditions[1]].append(mne.read_evokeds(cfg.fname.evokeds(subject=subject), condition=cfg.conditions[1]))
    evokeds_dict['Difference'].append(mne.read_evokeds(cfg.fname.evokeds(subject=subject), condition='Difference'))
    evokeds_dict['Total'].append(mne.read_evokeds(cfg.fname.evokeds(subject=subject), condition='Total'))

for key, _evokeds_list in evokeds_dict.items():
    _grand_average = mne.grand_average(_evokeds_list, interpolate_bads=True)
    _grand_average.comment = key
    grand_dict[key] = _grand_average

assert all([cfg.choi in g.ch_names for g in grand_dict.values()]), 'Channel to be plottet not in grand average of evokeds'
mne.write_evokeds(cfg.fname.grand_ave, list(grand_dict.values()), overwrite=True)

# ---------------------------------------------------------------------------------------------------------
# AMPLITUDE AND LATENCY DISTRIBUTION ACROSS EVOKEDS (SINGLE CHANNEL)
print('>>> Create dataframe from evokeds data')
df_max = pd.DataFrame(columns=['time', cfg.choi, 'condition'])
# Extract maximum amplitude and corresponding latency per evoked in time window of interest
for key, _evokeds_list in evokeds_dict.items():
    for _evoked in _evokeds_list:
        _evoked_choi = _evoked.copy().pick_channels([cfg.choi])
        _df_evoked = _evoked_choi.crop(cfg.test_tmin, cfg.test_tmax).to_data_frame()
        _df_row = _df_evoked.iloc[_df_evoked[cfg.choi].idxmax()]
        _df_row['condition'] = key
        df_max = df_max.append(_df_row)
        # df_max = pd.concat([df_max, _df_row], axis=0)

df_max.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------------------------------------------------------
# PERMUTATION T-TEST (ALL CHANNELS)
print('>>> Perform permutation T-test')
# Compute mean per evoked_diff
t_test_data = np.array([np.mean(_evoked.get_data(tmin=cfg.test_tmin, tmax=cfg.test_tmax), axis=1) for _evoked in evokeds_dict['Difference']])
info_obj = evokeds_dict['Difference'][0].copy().pick_types(meg=False, eeg=True, eog=False)
picks = mne.pick_types(info_obj.info, meg=False, eeg=True, eog=False)
fig_grand_topomap, fig_histogram_t_test  = utils.perform_t_test(t_test_data, info_obj, picks)

# ---------------------------------------------------------------------------------------------------------
# PERMUTATION CLUSTER TEST (SINGLE CHANNEL)
print('>>> Perform permutation cluster test')
data_cond1 = []
data_cond2 = []
for _evoked in evokeds_dict[cfg.conditions[0]]:
    data_cond1.append(_evoked.copy().pick_channels([cfg.choi]).get_data()[0,:])
for _evoked in evokeds_dict[cfg.conditions[1]]:
    data_cond2.append(_evoked.copy().pick_channels([cfg.choi]).get_data()[0,:])
  
permutation_test_data = [np.array(data_cond1), np.array(data_cond2)]
fig_permutation_cluster_test = utils.perform_permutation_cluster_test(permutation_test_data, grand_dict['Difference'])


# Add a plot of the data to the HTML report
with mne.open_report(cfg.fname.report_analysis_all) as report:
    report.title = 'Peak Analysis: All'
    # ---------------------------------------------------------------------------------------------------------
    fig_evoked = grand_dict['Total'].plot(ylim=dict(eeg=(-20, 20)), spatial_colors=True, show=False)
    # ---------------------------------------------------------------------------------------------------------
    times = np.arange(0, cfg.epochs_tmax, 0.1)
    fig_topomaps = [grand_dict[condition].plot_topomap(times=times, average=0.050, vmin=-20, vmax=20, show=False) for condition in cfg.conditions]
    # ---------------------------------------------------------------------------------------------------------

    # Compare ERPs of targets, distractors and difference of both
    kwargs = dict(picks=cfg.choi, truncate_yaxis=False, legend='lower right', show_sensors='upper right', show=False)
    fig_comparison = mne.viz.plot_compare_evokeds(grand_dict, **kwargs)
    # ---------------------------------------------------------------------------------------------------------
    html_peak_info_all, fig_peak_info_all = utils.get_peak_info(grand_dict['Difference'])
    html_peak_info_choi, fig_peak_info_choi = utils.get_peak_info(grand_dict['Difference'].pick_channels([cfg.choi]))
    # ---------------------------------------------------------------------------------------------------------

    fig_histogram_matrix, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 12))
    sns.histplot(df_max, x=cfg.choi, alpha = .7, hue='condition', bins=10, binrange=(0,50), hue_order=[cfg.conditions[1], cfg.conditions[0]], ax=ax[0,0])
    ax[0,0].set_title('Peak Amplitudes')
    sns.histplot(df_max, x='time', alpha = .7, hue='condition', bins=10, binrange=(0,0.8), hue_order=[cfg.conditions[1], cfg.conditions[0]], ax=ax[0,1])
    ax[0,1].set_title('Peak Latencies')
    sns.histplot(df_max, x=cfg.choi, y='time', bins=20, cbar=True, discrete=(False,False), ax=ax[1,0])
    ax[1,0].set_title('Bivariate histogram of peaks and latencies')
    sns.ecdfplot(df_max, x=cfg.choi, hue='condition', ax=ax[1,1])
    ax[1,1].set_title('Empirical cumulative distribution of peak amplitudes')
    # ---------------------------------------------------------------------------------------------------------
    utils.add_html(report, 'analyze_all.implementation','Task analyze_all', 'Analysis_all')

    title_left = 'Peak: all channels'
    title_right = 'Peak: choi' 
    fig_peak_measures = utils.combine_figures(fig_peak_info_all, title_left, fig_peak_info_choi, title_right)
    report.add_figure(fig_evoked, 'Evoked (Total)', replace=True, tags='Analysis_all')
    report.add_figure(fig_topomaps, title='Scalp topography maps', caption=cfg.conditions, replace=True, tags='Analysis_all')
    utils.add_html(report, 'analyze_all.comparison_grand','DESC: Comparison of grand averages ', 'Analysis_all')
    report.add_figure(fig_comparison,'Comparison of grand averages ({})'.format(cfg.choi), replace=True, tags='Analysis_all')
    report.add_figure(fig_peak_measures,'Peak measures', replace=True, tags='Analysis_all')
    report.add_html(html_peak_info_all,'Peak analysis: all channels', replace=True, tags='Analysis_all')
    report.add_html(html_peak_info_choi,'Peak analysis: {}'.format(cfg.choi), replace=True, tags='Analysis_all')
    utils.add_html(report, 'analyze_all.histograms','DESC: Amplitudes and Latencies across Epochs', 'Analysis_all')
    report.add_figure(fig_histogram_matrix, 'Amplitudes and Latencies across Subjects', replace=True, tags='Analysis_all')  
    utils.add_html(report, 'analyze_all.permutation_t_test','DESC: Permutation T-Test', 'Analysis_all')
    fig_permutation_t_test = utils.combine_figures(fig_grand_topomap, 'Statistical Topomap', fig_histogram_t_test, 'Histogram', fig_size=(16, 6))
    report.add_figure(fig_permutation_t_test, 'Permutation T-Test', replace=True, tags='Analysis_all')    
    utils.add_html(report, 'analyze_all.permutation_cluster_test','DESC: Permutation Cluster Test {}'.format(cfg.choi), 'Analysis_all')
    report.add_figure(fig_permutation_cluster_test,'Permutation Cluster Test ({})'.format(cfg.choi), replace=True, tags='Analysis_all')
    report.save(cfg.fname.report_analysis_all_html, overwrite=True, open_browser=False)

