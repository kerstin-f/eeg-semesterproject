
import config as cfg
import mne
mne.set_log_level(cfg.mne_log_level)
import os
import os.path

if not os.path.isfile(cfg.fname.volume_source_space):
    fsaverage_dir = mne.datasets.fetch_fsaverage(verbose=cfg.mne_log_level)
    fsaverage_bem = os.path.join(fsaverage_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    fsaverage_mri = os.path.join(fsaverage_dir, 'mri', 'T1.mgz')

    vol_grid_mm = 3.0
    vol_src = mne.setup_volume_source_space(
        'fsaverage', mri=fsaverage_mri, pos=vol_grid_mm, bem=fsaverage_bem, add_interpolator=True, verbose=cfg.mne_log_level)
    vol_src.save(cfg.fname.volume_source_space, overwrite=True)


