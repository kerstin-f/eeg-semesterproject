"""
Source localization visualization via screenshots of the interactive mne.SourceEstimate tool.

Splitted into seperate task to not be forced to regenerate the data when playing around with the plots.
"""

import utils
import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level) #cfg.mne_log_level) 
import numpy as np
from matplotlib import pyplot as plt
import os

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# ---------------------------------------------------------------------------------------------------------

# Requires modules pyvistaqt PyQt5
mne.viz.set_3d_backend('pyvistaqt')

# Download fsaverage files to mne_data\MNE-fsaverage-data\fsaverage\bem
fsaverage_dir = mne.datasets.fetch_fsaverage(verbose=cfg.mne_log_level)
fsaverage_root_dir = os.path.join(fsaverage_dir, '..')
fsaverage_src = os.path.join(fsaverage_dir, 'bem', 'fsaverage-ico-5-src.fif')
fsaverage_bem = os.path.join(fsaverage_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
fsaverage_mri = os.path.join(fsaverage_dir, 'mri', 'T1.mgz')
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation

# ---------------------------------------------------------------------------------------------------------

print('>> Read epochs')
epochs = mne.read_epochs(cfg.fname.epoched(subject=subject), preload=True)
epochs.pick_types(eog=False, eeg=True)
# epochs.interpolate_bads()
epochs.set_eeg_reference('average', projection=True)

print('>> Read evoked')
evoked_total = mne.read_evokeds(cfg.fname.evokeds(subject=subject), condition='Total').set_eeg_reference('average', projection=True)

print('>> Read noise covariance')
noise_cov = mne.read_cov(cfg.fname.noise_covariance(subject=subject))

print('>> Read data covariance')
data_cov = mne.read_cov(cfg.fname.data_covariance(subject=subject))

print('>> Read forward solution (surface)')
fwd_sur = mne.read_forward_solution(cfg.fname.forward_solution_surface(subject=subject))

print('>> Read source estimate (evoked total, surface)')
stc_evoked_total_sur = mne.read_source_estimate(cfg.fname.source_estimate_evoked_total_surface(subject=subject), subject=trans)
_, time_peak_total = stc_evoked_total_sur.get_peak()
vertno_peak_lh_total, time_peak_lh_total = stc_evoked_total_sur.get_peak(hemi='lh')
vertno_peak_rh_total, time_peak_rh_total = stc_evoked_total_sur.get_peak(hemi='rh')

print('>> Read source estimate (evoked target, surface)')
stc_evoked_target_sur = mne.read_source_estimate(cfg.fname.source_estimate_evoked_target_surface(subject=subject), subject=trans)
_, time_peak_target = stc_evoked_target_sur.get_peak()
vertno_peak_lh_target, time_peak_lh_target = stc_evoked_target_sur.get_peak(hemi='lh')
vertno_peak_rh_target, time_peak_rh_target = stc_evoked_target_sur.get_peak(hemi='rh')

print('>> Read source estimate (evoked distractor, surface)')
stc_evoked_distractor_sur = mne.read_source_estimate(cfg.fname.source_estimate_evoked_distractor_surface(subject=subject), subject=trans)
_, time_peak_distractor = stc_evoked_distractor_sur.get_peak()
vertno_peak_lh_distractor, time_peak_lh_distractor = stc_evoked_distractor_sur.get_peak(hemi='lh')
vertno_peak_rh_distractor, time_peak_rh_distractor = stc_evoked_distractor_sur.get_peak(hemi='rh')

print('>> Read source estimate (evoked difference, surface)')
stc_evoked_diff_sur = mne.read_source_estimate(cfg.fname.source_estimate_evoked_difference_surface(subject=subject), subject=trans)
_, time_peak_diff = stc_evoked_diff_sur.get_peak()
vertno_peak_lh_diff, time_peak_lh_diff = stc_evoked_diff_sur.get_peak(hemi='lh')
vertno_peak_rh_diff, time_peak_rh_diff = stc_evoked_diff_sur.get_peak(hemi='rh')

print('>> Read source estimate (evoked total, volume)')
stc_evoked_total_vol = mne.read_source_estimate(cfg.fname.source_estimate_evoked_total_volume(subject=subject) + '-vl.stc', subject=trans)
vertno_vol, time_peak_vol = stc_evoked_total_vol.get_peak()

print('>> Read volume source space')
vol_src = mne.read_source_spaces(cfg.fname.volume_source_space)
# ---------------------------------------------------------------------------------------------------------

if cfg.generate_reports:
    with mne.open_report(cfg.fname.report_source_localization(subject=subject)) as report:
        
        report.title = 'Source Localization: Subject ' + subject
        
        custom_css="""
        pre { margin-bottom: -20px; }
        """
        report.add_custom_css(custom_css)

        # Helper
        def add_brain_screenshot(report, brain, title):
            fig_brain = brain.screenshot(time_viewer=True)
            report.add_figure(fig_brain, title)
            brain.close()

        def add_brain_screenshot_range(report, brain, time_range, title):
            figs_brain = []
            for t in time_range:
                brain.set_time(t)
                #brain.setup_time_viewer()
                brain_sur_kwargs['initial_time'] = t
                brain.plot_time_line()
                figs_brain.append(brain.screenshot(time_viewer=True))
                
            report.add_figure(figs_brain, title)
            brain.clear_glyphs()
            brain.close()

        # Plot parameters
        brain_sur_kwargs = dict(size=(1000,600), initial_time=time_peak_total, time_unit='s', time_viewer=True, show_traces=True, background='white')
        distance = 280
        
        brain_sur_kwargs['clim'] = dict(kind='value', lims=[0.0, 40.0, 85.0])
        brain_sur_kwargs['hemi'] = 'split'
        brain_sur_kwargs['smoothing_steps'] = 5
        brain_sur_kwargs['cortex'] = 'classic'
        brain_sur_kwargs['views'] = ['lateral','medial']
        brain_sur_kwargs['surface'] = 'white'

        brain_vol_kwargs = dict(subject=trans, show=False)
        brain_vol_kwargs['clim'] = dict(kind='value', lims=[0.0, 100.0, 210.0])

        num_screenshots = 11
        t_pm = 0.2
        time_range_total = np.linspace(time_peak_total - t_pm, time_peak_total + t_pm, num=num_screenshots)
        time_range_target = np.linspace(time_peak_target - t_pm, time_peak_target + t_pm, num=num_screenshots)
        time_range_distractor = np.linspace(time_peak_distractor - t_pm, time_peak_distractor + t_pm, num=num_screenshots)
        time_range_diff = np.linspace(time_peak_diff - t_pm, time_peak_diff + t_pm, num=num_screenshots)
        time_range_vol = np.linspace(time_peak_vol - t_pm, time_peak_vol + t_pm, num=num_screenshots)

        # ---------------------------------------------------------------------------------------------------------

        utils.add_html(report, 'source_localization.implementation',
                       'Task source_localization', 'SourceLocalization')
                       
        print('>>> Plot noise covariance matrix')
        fig_cov, fig_spec  = mne.viz.plot_cov(noise_cov, epochs.info, show=False)
        fig_cov_combined = utils.combine_figures(fig_cov, 'Noise covariance matrix', fig_spec, 'Spectra of the noise covariance')
        report.add_figure(fig_cov_combined, 'Noise covariance')

        # ---------------------------------------------------------------------------------------------------------

        utils.add_html(report, 'source_localization.topomap_compare_covariance',
                       'DESC: Covariance topomap comparison', 'SourceLocalization')
                       
        print('>>> Plot topomap comparison')
        fig_topomap_noise_cov = noise_cov.plot_topomap(epochs.info)
        fig_topomap_data_cov = data_cov.plot_topomap(epochs.info)
        fig_topomap_data_whitened = data_cov.plot_topomap(epochs.info, noise_cov=noise_cov)
        fig_topomaps_combined = utils.combine_figures(
            fig_topomap_noise_cov, 'Noise', 
            fig_topomap_data_cov, 'Data',
            fig_topomap_data_whitened, 'Whitened', fig_size=(16, 6))

        report.add_figure(fig_topomaps_combined, 'Covariance topomap comparison')

        # ---------------------------------------------------------------------------------------------------------

        utils.add_html(report, 'source_localization.source_estimates',
                        'DESC: Source estimate', 'SourceLocalization')

        title = 'Source estimate (total)'
        print('>>> Brain plot', title)
        brain = stc_evoked_total_sur.plot(**brain_sur_kwargs)
        #brain.add_foci(vertno_peak_lh_total, coords_as_verts=True, hemi='lh', color='blue', scale_factor=0.6, alpha=0.5)
        #brain.add_foci(vertno_peak_rh_total, coords_as_verts=True, hemi='rh', color='red', scale_factor=0.6, alpha=0.5)
        brain.show_view(distance=distance)
        add_brain_screenshot_range(report, brain, time_range_total, title)

        # ---------------------------------------------------------------------------------------------------------

        title = 'Source estimate (distractor)'
        print('>>> Brain plot', title)
        brain = stc_evoked_distractor_sur.plot(**brain_sur_kwargs)
        #brain.add_foci(vertno_peak_lh_distractor, coords_as_verts=True, hemi='lh', color='blue', scale_factor=0.6, alpha=0.5)
        #brain.add_foci(vertno_peak_rh_distractor, coords_as_verts=True, hemi='rh', color='red', scale_factor=0.6, alpha=0.5)
        brain.show_view(distance=distance)
        add_brain_screenshot_range(report, brain, time_range_distractor, title)

        # ---------------------------------------------------------------------------------------------------------
                       
        title = 'Source estimate (target)'
        print('>>> Brain plot', title)
        brain = stc_evoked_target_sur.plot(**brain_sur_kwargs)
        #brain.add_foci(vertno_peak_lh_target, coords_as_verts=True, hemi='lh', color='blue', scale_factor=0.6, alpha=0.5)
        #brain.add_foci(vertno_peak_rh_target, coords_as_verts=True, hemi='rh', color='red', scale_factor=0.6, alpha=0.5)
        brain.show_view(distance=distance)
        add_brain_screenshot_range(report, brain, time_range_target, title)
        
        # ---------------------------------------------------------------------------------------------------------

        title = 'Source estimate (difference)'
        print('>>> Brain plot', title)
        brain_sur_kwargs['clim'] = dict(kind='value', lims=[0.0, 30.0, 40.0])
        brain = stc_evoked_diff_sur.plot(**brain_sur_kwargs)
        #brain.add_foci(vertno_peak_lh_diff, coords_as_verts=True, hemi='lh', color='blue', scale_factor=0.6, alpha=0.5)
        #brain.add_foci(vertno_peak_rh_diff, coords_as_verts=True, hemi='rh', color='red', scale_factor=0.6, alpha=0.5)
        brain.show_view(distance=distance)
        add_brain_screenshot_range(report, brain, time_range_diff, title)

        # ---------------------------------------------------------------------------------------------------------

        if subject==cfg.create_video_for_subject:

            utils.add_html(report,'source_localization.movie',
                            'DESC: Source estimate movie', 'SourceLocalization')
                            
            time_dilation = 50
            tmin=0.2
            tmax=0.7
            framerate=30

            brain_sur_kwargs = dict(size=(1600, 800), hemi= 'both', surface='white',cortex='high_contrast', initial_time=0, time_unit='s', time_viewer=True, show_traces=True, background='white')
            brain_sur_kwargs['clim'] = dict(kind='value', lims=[0.0, 30.0, 40.0])

            print('>>> Source localization movie')
            brain_sur_kwargs['views'] = ['dorsal', 'ventral']
            filename = cfg.fname.source_movie_evoked_difference(subject=subject)
            path='./reports/' + filename
            brain = stc_evoked_diff_sur.plot(**brain_sur_kwargs)
            brain.save_movie(filename=path, time_dilation=time_dilation, tmin=tmin, tmax=tmax, framerate=framerate, interpolation='linear', time_viewer=True)
            brain.close()
            embedded_movie_html = """<center><video controls id="example_video_1" class="video-js vjs-default-skin" width="1600" height="800" src="./""" + filename + """" type='video/mp4' /></video></center>"""
            report.add_html(embedded_movie_html, 'Source localization movie (evoked difference)')


        # ---------------------------------------------------------------------------------------------------------

        utils.add_html(report,'source_localization.volume',
                        'DESC: Source estimate volume', 'SourceLocalization')
                        
        title = 'Source estimate volume (mode=glass_brain)'
        print('>>> Brain plot ', title)
        figs_vol = []
        plt.rcParams["figure.figsize"] = (10, 6)
        for t in time_range_vol:
            print('t={:.2f}s '.format(t), end='')
            brain_vol_kwargs['initial_time'] = t
            figs_vol.append(stc_evoked_total_vol.plot(vol_src, mode='glass_brain', **brain_vol_kwargs))
        print('')
        
        report.add_figure(figs_vol, title)

        report.save(cfg.fname.report_source_localization_html(subject=subject), overwrite=True, open_browser=False)


        # ================================================================
        # print('>> Plot alignment')
        # plotter_alignment = mne.viz.plot_alignment(
        #     epochs.info, src=fsaverage_src, coord_frame='mri', ecog=False, seeg=False, fnirs=False, dbs=False, trans=trans, eeg=['original', 'projected'],
        #     show_axes=False, surfaces=dict(brain=0.5, head=0.5), verbose=cfg.mne_log_level) 

        # mne.viz.set_3d_view(figure=plotter_alignment, azimuth=0, elevation=0, distance=0.6)
        # fig_alignment1 = plotter_alignment.plotter.screenshot()
        # mne.viz.set_3d_view(figure=plotter_alignment, azimuth=270, elevation=90, distance=0.6)
        # fig_alignment2 = plotter_alignment.plotter.screenshot()
        # mne.viz.set_3d_view(figure=plotter_alignment, azimuth=90, elevation=90, distance=0.6)
        # fig_alignment3 = plotter_alignment.plotter.screenshot()
        # mne.viz.set_3d_view(figure=plotter_alignment, azimuth=180, elevation=90, distance=0.6)
        # fig_alignment3 = plotter_alignment.plotter.screenshot()

        # plotter_alignment.plotter.close()
        # report.add_figure(fig_alignment1, 'Alignment 0, 0')
        # report.add_figure(fig_alignment2, 'Alignment 270, 90')
        # report.add_figure(fig_alignment3, 'Alignment 90, 90')

        # ================================================================
        # print('>> Plot joined')
        # fig_plot_joint_target = evokeds_target.plot_joint(show=False)
        # fig_plot_joint_distractor = evokeds_distractor.plot_joint(show=False)
        # report.add_figure(fig_plot_joint_target, 'Evoked with peak Topomaps (Target)')
        # report.add_figure(fig_plot_joint_distractor, 'Evoked with peak Topomaps (Distractor)')

        # ================================================================
        # print('>> Plot topomaps')
        # plt.figure()
        # num_plt_h = 10
        # num_plt_v = 3
        # num_topomaps = num_plt_h*num_plt_v
        # step_topomap = 5
        # offset_topomap = 5000
        # for d in range(3):
        #     fig_topomap_d, ax = plt.subplots(num_plt_v,num_plt_h, figsize=(15, 5))
        #     n = 0
        #     for i in range(0,num_plt_v):
        #         for j in range(0,num_plt_h):
        #             mne.viz.plot_topomap(fwd_sur["sol"]["data"][:,offset_topomap+n*step_topomap*3+d], epochs.info, axes=ax[i,j], show=False)
        #             n+=1
        #     report.add_figure(fig_topomap_d, 'Topomaps {}'.format(d))

        # ================================================================


