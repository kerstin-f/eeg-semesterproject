"""
Perform bandpass filtering.
"""
import os
import config as cfg

dir = cfg.fname.reports_dir
filelist = [ f for f in os.listdir(dir)]
for f in filelist:
    os.remove(os.path.join(dir, f))