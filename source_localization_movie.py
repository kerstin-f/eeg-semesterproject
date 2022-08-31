
import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level) #cfg.mne_log_level) 
import numpy as np
from matplotlib import pyplot as plt
import os

# Requires imageio, imageio-ffmpeg, pyvistaqt, PyQt5

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject


print('Processing subject:', subject)

# Requires modules 
mne.viz.set_3d_backend('pyvistaqt')

trans = 'fsaverage'

print('>> Read source estimate (evoked, surface)')
stc_evokeds_sur = mne.read_source_estimate(cfg.fname.source_estimate_evoked_total_surface(subject=subject), subject=trans)
_, time_peak = stc_evokeds_sur.get_peak()
vertno_peak_lh, time_peak_lh = stc_evokeds_sur.get_peak(hemi='lh')
vertno_peak_rh, time_peak_rh = stc_evokeds_sur.get_peak(hemi='rh')

report = mne.Report(title='Movie')

time_dilation = 1
tmin=-0.15
tmax=0.75
framerate=30

brain_sur_kwargs = dict(size=(1200, 1000), hemi= 'both', surface='white',cortex='high_contrast', initial_time=time_peak, time_unit='s', time_viewer=True, show_traces=True, background='white')
brain_sur_kwargs['clim'] = dict(kind='value', lims=[0.0, 40.0, 100.0])

print('>>> Source localization movie')
brain_sur_kwargs['views'] = ['dorsal', 'ventral']
filename = 'source_localization.mp4'
path='./reports/' + filename
brain = stc_evokeds_sur.plot(**brain_sur_kwargs)
brain.save_movie(filename=path, time_dilation=time_dilation, tmin=tmin, tmax=tmax, framerate=framerate, interpolation='linear', time_viewer=True)
brain.close()
embedded_movie_html = """<center><video controls id="example_video_1" class="video-js vjs-default-skin" width="800" height="750" src="./""" + filename + """" type='video/mp4' /></video></center>"""
report.add_html(embedded_movie_html, 'Source localization movie (lateral)')


report.save(cfg.fname.source_localization_movie_html(subject=subject), overwrite=True, open_browser=False)
