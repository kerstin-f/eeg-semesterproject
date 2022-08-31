
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

print('')
print('===================')
print('Source localization')
print('===================')
print('')
print('Processing subject:', subject)

# Requires modules pyvistaqt PyQt5
mne.viz.set_3d_backend('pyvistaqt')

trans = 'fsaverage'

print('>> Read epochs')
epochs = mne.read_epochs(cfg.fname.epoched(subject=subject), preload=True)

print('>> Read noise covariance')
noise_cov = mne.read_cov(cfg.fname.noise_covariance(subject=subject))

print('>> Read forward solution (surface)')
fwd_sur = mne.read_forward_solution(cfg.fname.forward_solution_surface(subject=subject))

print('>> Read source estimate (evoked, surface)')
stc_evokeds_sur = mne.read_source_estimate(cfg.fname.source_estimate_evoked_total_surface(subject=subject), subject=trans)
_, time_peak = stc_evokeds_sur.get_peak()
vertno_peak_lh, time_peak_lh = stc_evokeds_sur.get_peak(hemi='lh')
vertno_peak_rh, time_peak_rh = stc_evokeds_sur.get_peak(hemi='rh')


report = mne.Report(title='Cheat sheets')

brain_sur_kwargs = dict(size=(800, 800), initial_time=time_peak, time_unit='s', time_viewer=False, show_traces=False, colorbar=False, time_label=None, background='white')

# ================================================================
print('>> Cheat sheet for "surface"')
plt.figure()

num_plt_v = 2
num_plt_h = 2
fig_params, ax_arr = plt.subplots(num_plt_v, num_plt_h, figsize=(10, 10), layout='tight')
plt.subplots_adjust(wspace=0, hspace=0)
args =  ['inflated', 'white', 'sphere']
n = 0
for i in range(0,num_plt_v):
    for j in range(0,num_plt_h):
        ax = ax_arr[i,j]
        ax.set_axis_off()
        if n < len(args):
            arg = args[n]
            n+=1
            title = 'view={}'.format(arg)
            brain = stc_evokeds_sur.plot(surface=arg, hemi='both', views='frontal', **brain_sur_kwargs)
            screenshot = brain.screenshot()
            ax.title.set_text(title)
            brain.close()
            ax.imshow(screenshot)

report.add_figure(fig_params, 'Cheat sheet for "surface"')

brain_sur_kwargs['surface'] = 'white'

# ================================================================
print('>> Cheat sheet for "hemi"')
plt.figure()
num_plt_v = 2
num_plt_h = 2
fig_params, ax_arr = plt.subplots(num_plt_v, num_plt_h, figsize=(10, 10), layout='tight')
plt.subplots_adjust(wspace=0, hspace=0)
args = ['lh', 'rh', 'both', 'split']
n = 0
for i in range(0,num_plt_v):
    for j in range(0,num_plt_h):
        ax = ax_arr[i,j]
        arg = args[n]
        n+=1
        title = 'hemi={}'.format(arg)
        brain = stc_evokeds_sur.plot(hemi=arg, views='frontal', **brain_sur_kwargs)
        screenshot = brain.screenshot()
        ax.title.set_text(title)
        ax.set_axis_off()
        ax.imshow(screenshot)
        brain.close()
        
report.add_figure(fig_params, 'Cheat sheet for "hemi"')

brain_sur_kwargs['hemi'] = 'both'

# ================================================================
print('>> Cheat sheet for "cortex"')
plt.figure()
num_plt_v = 2
num_plt_h = 2
fig_params, ax_arr = plt.subplots(num_plt_v, num_plt_h, figsize=(10, 10), layout='tight')
plt.subplots_adjust(wspace=0, hspace=0)
args =  ['classic', 'bone', 'low_contrast', 'high_contrast']
n = 0
for i in range(0,num_plt_v):
    for j in range(0,num_plt_h):
        ax = ax_arr[i,j]
        ax.set_axis_off()
        if n < len(args):
            arg = args[n]
            n+=1
            title = 'view={}'.format(arg)
            brain = stc_evokeds_sur.plot(cortex=arg, views='frontal', **brain_sur_kwargs)
            screenshot = brain.screenshot()
            ax.title.set_text(title)
            brain.close()
            ax.imshow(screenshot)

report.add_figure(fig_params, 'Cheat sheet for "cortex"')

brain_sur_kwargs['cortex'] = 'classic'
brain_sur_kwargs['hemi'] = 'lh'

# ================================================================
print('>> Cheat sheet for "views"')
plt.figure()
num_plt_v = 3
num_plt_h = 4
fig_params, ax_arr = plt.subplots(num_plt_v, num_plt_h, figsize=(10, 10), layout='tight')
plt.subplots_adjust(wspace=0, hspace=0)
args =  ['lateral', 'medial', 'rostral', 'caudal', 'dorsal', 'ventral', 'frontal', 'parietal', 'axial', 'sagittal', 'coronal']
n = 0
for i in range(0,num_plt_v):
    for j in range(0,num_plt_h):
        ax = ax_arr[i,j]
        ax.set_axis_off()
        if n < len(args):
            arg = args[n]
            n+=1
            title = 'view={}'.format(arg)
            brain = stc_evokeds_sur.plot(views=arg, **brain_sur_kwargs)
            screenshot = brain.screenshot()
            ax.title.set_text(title)
            brain.close()
            ax.imshow(screenshot)

report.add_figure(fig_params, 'Cheat sheet for "views"')

# ================================================================
# ================================================================

report.save(cfg.fname.source_localization_cheatsheet_html, overwrite=True, open_browser=False)







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
        # print('>> Plot covariance t=[-1s..0s]')
        # fig_cov, fig_spec  = mne.viz.plot_cov(noise_cov, epochs.info, show=False)
        # report.add_figure(fig_cov, 'Covariance')

        # ================================================================
        # if calc_volume:
        #     print('>> Plot source estimates (volume)')

        #     brain_vol_kwargs = dict(subject='fsaverage', show=False, initial_time=time_peak)

        #     # title = 'Source estimate volume (defaults)'
        #     # print('>>> Brain plot ', title)
        #     # fig_vol = stc_vol.plot(vol_src, **brain_vol_kwargs)
        #     # report.add_figure(fig_vol, title)

        #     title = 'Source estimate volume (mode=glass_brain)'
        #     print('>>> Brain plot ', title)
        #     fig_vol = stc_vol.plot(vol_src, mode='glass_brain', **brain_vol_kwargs)
        #     report.add_figure(fig_vol, title)

        # ================================================================

        # # Helper
        # def add_brain_screenshot(report, brain, title):
        #     fig_brain = brain.screenshot(time_viewer=True)
        #     report.add_figure(fig_brain, title)
        #     brain.close()

        # brain_sur_kwargs = dict(size=(400, 400), initial_time=time_peak, time_unit='s', time_viewer=False, show_traces=False, background='white')


        # for arg in ['classic', 'bone', 'low_contrast', 'high_contrast']:
        #     title = 'cortex=' + str(arg)
        #     print('>>> Brain plot', title)
        #     brain = stc_evokeds_sur.plot(cortex=arg, **brain_sur_kwargs)
        #     add_brain_screenshot(report, brain, title)



        # for arg in ['lateral', 'medial', 'rostral', 'caudal', 'dorsal', 'ventral', 'frontal', 'parietal', 'axial', 'sagittal', 'coronal']:
        #     title = 'views=' + str(arg)
        #     print('>>> Brain plot', title)
        #     brain = stc_evokeds_sur.plot(views=arg, **brain_sur_kwargs)
        #     add_brain_screenshot(report, brain, title)


        # title = 'Source estimate surface (defaults)'
        # print('>>> Brain plot ', title)
        # brain = stc_evokeds_sur.plot(**brain_sur_kwargs)
        # add_brain_screenshot(report, brain, title)

        # for arg in ['inflated', 'white']: # 'sphere'
        #     title = 'surface=' + str(arg)
        #     print('>>> Brain plot', title)
        #     brain = stc_evokeds_sur.plot(surface=arg, **brain_sur_kwargs)
        #     add_brain_screenshot(report, brain, title)


        # for arg in ['lh', 'rh', 'both', 'split']:
        #     title = 'hemi=' + str(arg)
        #     print('>>> Brain plot', title)
        #     brain = stc_evokeds_sur.plot(hemi=arg, **brain_sur_kwargs)
        #     add_brain_screenshot(report, brain, title)

        # brain_sur_kwargs['hemi'] = 'lh'

        # # ================================================================
        # print('>>> Brain video')
        # brain_sur_kwargs['size'] = (1600, 800)
        # brain_sur_kwargs['hemi'] = 'split'
        # brain_sur_kwargs['time_viewer'] = True
        # brain_sur_kwargs['show_traces'] = True
        # brain = stc_evokeds_sur.plot(**brain_sur_kwargs)
        # brain.add_foci(vertno_peak_lh, coords_as_verts=True, hemi='lh', color='blue', scale_factor=0.6, alpha=0.5)
        # brain.add_foci(vertno_peak_rh, coords_as_verts=True, hemi='rh', color='red', scale_factor=0.6, alpha=0.5)

        # # Requires imageio, imageio-ffmpeg
        # brain.save_movie(filename='test.mp4', time_dilation=2, tmin=0.0, tmax=1.0, framerate=10,
        #                  interpolation='linear', time_viewer=True)

        # embedded_movie_html = """<center><video controls id="example_video_1" class="video-js vjs-default-skin" width="640" height="264" src="file:///C:/Users/kerst/git/eeg-semesterproject/test.mp4" type='video/mp4' /></video></center>"""
        # report.add_html(embedded_movie_html, "Brain video")
        # brain.close()
        # brain_sur_kwargs['show_traces'] = False
        # brain_sur_kwargs['time_viewer'] = False
        # brain_sur_kwargs['hemi'] = 'lh'
        # brain_sur_kwargs['size'] = (800, 400)
        # # ================================================================

        # for arg in [0.1, 0.4, 0.7, 1.0]:
        #     title = 'alpha=' + str(arg)
        #     print('>>> Brain plot ', title)
        #     brain = stc_evokeds_sur.plot(alpha=arg, **brain_sur_kwargs)
        #     add_brain_screenshot(report, brain, title)

        # brain_sur_kwargs['alpha'] = 1.0

        # for arg in [1, 3, 5]:
        #     title = 'smoothing_steps=' + str(arg)
        #     print('>>> Brain plot ', title)
        #     brain = stc_evokeds_sur.plot(smoothing_steps=arg, **brain_sur_kwargs)
        #     add_brain_screenshot(report, brain, title)

        # brain_sur_kwargs['smoothing_steps'] = 5

        # for arg in ['classic', 'bone', 'low_contrast', 'high_contrast']:
        #     title = 'cortex=' + str(arg)
        #     print('>>> Brain plot', title)
        #     brain = stc_evokeds_sur.plot(cortex=arg, **brain_sur_kwargs)
        #     add_brain_screenshot(report, brain, title)

        # brain_sur_kwargs['cortex'] = 'classic'

        # for arg in ['lateral', 'medial', 'rostral', 'caudal', 'dorsal', 'ventral', 'frontal', 'parietal', 'axial', 'sagittal', 'coronal']:
        #     title = 'views=' + str(arg)
        #     print('>>> Brain plot', title)
        #     brain = stc_evokeds_sur.plot(views=arg, **brain_sur_kwargs)
        #     add_brain_screenshot(report, brain, title)

        # brain_sur_kwargs['views'] = 'lateral'
            
        # for arg in [100, 200, 300, 400, 500]:
        #     title = 'distance=' + str(arg)
        #     print('>>> Brain plot', title)
        #     brain = stc_evokeds_sur.plot(**brain_sur_kwargs)
        #     brain.show_view(distance=arg)
        #     add_brain_screenshot(report, brain, title)

        # view_kwargs = {}
        # view_kwargs['distance'] = 400

        # for arg in [0, 90, 180, 270]:
        #     title = 'azimuth=' + str(arg)
        #     print('>>> Brain plot', title)
        #     brain = stc_evokeds_sur.plot(**brain_sur_kwargs)
        #     brain.show_view(azimuth=arg)
        #     add_brain_screenshot(report, brain, title)

        # for arg in [0, 90, 180, 270]:
        #     title = 'elevation=' + str(arg)
        #     print('>>> Brain plot', title)
        #     brain = stc_evokeds_sur.plot(**brain_sur_kwargs)
        #     brain.show_view(elevation=arg)
        #     add_brain_screenshot(report, brain, title)

        # title = 'add head set view'
        # print('>>> Brain plot', title)
        # brain_sur_kwargs['hemi'] = 'both'
        # brain = stc_evokeds_sur.plot(**brain_sur_kwargs)
        # brain.add_head()
        # brain.show_view(azimuth=0, elevation=0, distance=600)
        # add_brain_screenshot(report, brain, title)


        # brain_sur_kwargs['hemi'] = 'lh'

        # # ================================================================
        # print('>>> Brain plot (split) at time of peak with alpha')
        # brain = stc.plot(size=(800, 400), surface='inflated', views="lateral", initial_time=time_peak, 
        #          hemi='split', alpha=0.5)
        # # brain.add_foci(vertno_peak, coords_as_verts=True, hemi='rh', color='blue',
        # #             scale_factor=0.6)
        # fig_brain = brain.screenshot()
        # report.add_figure(fig_brain, 'Source estimates with alpha')
        # brain.close()


        # # ================================================================
        # print('>>> Brain plot defaults')
        # brain = stc.plot()
        # fig_brain = brain.screenshot()
        # report.add_figure(fig_brain, 'Source estimates low_contrast')
        # brain.close()


    #     # ================================================================
    #     print('>>> Brain plot (split) at time of peak')
    #     #brain = stc.plot(size=(800, 400), surface='inflated', views="lateral", initial_time=time_peak, hemi='split', background='white')
    #     brain = stc.plot(surface='inflated', views="lateral", hemi='split', **brain_sur_kwargs)
    #     fig_brain = brain.screenshot()
    #     report.add_figure(fig_brain, 'Source estimates')
    #     brain.close()

    #     # ================================================================
    #     print('>>> Brain plot surface')
    #     brain = stc.plot(surface='inflated')
    #     fig_brain = brain.screenshot()
    #     report.add_figure(fig_brain, 'Source estimates low_contrast')
    #     brain.close()

    #     # ================================================================
    #     print('>>> Brain plot with label')
    #     brain = stc.plot(size=(800, 400), alpha=0.1, background='white', cortex='low_contrast')
    #     brain.add_label('BA44', hemi='lh', color='green', borders=True)
    #     brain.show_view(azimuth=190, elevation=70, distance=350, focalpoint=(0, 0, 20))
    #     fig_brain = brain.screenshot()
    #     report.add_figure(fig_brain, 'Source estimates with label ')
    #     brain.close()

    #    # ================================================================
    #     print('>>> Brain plot with head')
    #     brain = stc.plot(size=(800, 400),initial_time=time_peak, background='white')
    #     brain.add_head(alpha=0.5)
    #     brain.reset_view()
    #     fig_brain = brain.screenshot()
    #     report.add_figure(fig_brain, 'Source estimates with head')
    #     brain.close()



        #brain.save_movie(time_dilation=20, tmin=0.05, tmax=0.16, interpolation='linear', framerate=10)
        # You can save a brain movie with:
        # brain.save_movie(time_dilation=20, tmin=0.05, tmax=0.16, framerate=10,
        #                  interpolation='linear', time_viewer=True)
