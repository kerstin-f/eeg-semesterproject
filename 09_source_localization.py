"""
Source estimate of evokeds (Total, Target, Distractor) for surface-based and volumetric source space.
There are no individual head models in the P3 data, so the FreeSurver average subject is used as a surrogate.
"""

import config as cfg
import argparse
import mne
mne.set_log_level(cfg.mne_log_level)
import os
import os.path


# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# ---------------------------------------------------------------------------------------------------------
# Download fsaverage files to mne_data\MNE-f
# average-data\fsaverage\bem
if not os.path.isfile(cfg.fname.volume_source_space):
    fsaverage_dir = mne.datasets.fetch_fsaverage(verbose=cfg.mne_log_level)
    fsaverage_bem = os.path.join(fsaverage_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    fsaverage_mri = os.path.join(fsaverage_dir, 'mri', 'T1.mgz')

    vol_grid_mm = 3.0
    vol_src = mne.setup_volume_source_space(
        'fsaverage', mri=fsaverage_mri, pos=vol_grid_mm, bem=fsaverage_bem, add_interpolator=True, verbose=cfg.mne_log_level)
    vol_src.save(cfg.fname.volume_source_space, overwrite=True)

# ---------------------------------------------------------------------------------------------------------
# Download fsaverage files to mne_data\MNE-fsaverage-data\fsaverage\bem
fsaverage_dir = mne.datasets.fetch_fsaverage(verbose=cfg.mne_log_level)
fsaverage_root_dir = os.path.join(fsaverage_dir, '..')
fsaverage_src = os.path.join(fsaverage_dir, 'bem', 'fsaverage-ico-5-src.fif')
fsaverage_bem = os.path.join(fsaverage_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
fsaverage_mri = os.path.join(fsaverage_dir, 'mri', 'T1.mgz')
trans = 'fsaverage'  # mne has a built-in fsaverage transformation

# ---------------------------------------------------------------------------------------------------------
print('>> Read epochs')
epochs = mne.read_epochs(cfg.fname.epoched(subject=subject), preload=True)
epochs.pick_types(eog=False, eeg=True)
epochs.apply_baseline(cfg.baseline)
# epochs.interpolate_bads()
epochs.set_eeg_reference('average', projection=True)
# epochs.equalize_event_counts() # No effect

print('>> Read evokeds')
evoked_distractor = mne.read_evokeds(cfg.fname.evokeds(subject=subject), condition='Distractor').set_eeg_reference('average', projection=True)
evoked_distractor.interpolate_bads()
evoked_target = mne.read_evokeds(cfg.fname.evokeds(subject=subject), condition='Target').set_eeg_reference('average', projection=True)
evoked_target.interpolate_bads()
evoked_diff = mne.read_evokeds(cfg.fname.evokeds(subject=subject), condition='Difference').set_eeg_reference('average', projection=True)
evoked_diff.interpolate_bads()
evoked_total = mne.read_evokeds(cfg.fname.evokeds(subject=subject), condition='Total').set_eeg_reference('average', projection=True)
evoked_total.interpolate_bads()

print('>> Compute noise covariance')
noise_cov = mne.compute_covariance(epochs,tmin=cfg.epochs_tmin, tmax=0.0, method='auto')
mne.write_cov(cfg.fname.noise_covariance(subject=subject), noise_cov, overwrite=True)

print('>> Compute data covariance')
data_cov = mne.compute_covariance(epochs, tmin=0.0, tmax=cfg.epochs_tmax, method='auto')
mne.write_cov(cfg.fname.data_covariance(subject=subject), data_cov, overwrite=True)

# ---------------------------------------------------------------------------------------------------------
# Surface
lambda2_sur = 1.0 / cfg.snr_sur ** 2

print('>> Make forward solution (surface)')
fwd_sur = mne.make_forward_solution(
    epochs.info, trans=trans, src=fsaverage_src, bem=fsaverage_bem, eeg=True, mindist=5.0, n_jobs=8)
mne.write_forward_solution(cfg.fname.forward_solution_surface(subject=subject), fwd_sur, overwrite=True)

print('>> Compute inverse operator (surface)')
inv_sur = mne.minimum_norm.make_inverse_operator(epochs.info, fwd_sur, noise_cov, verbose=cfg.mne_log_level, loose=0.2, depth=0.8)
# mne.minimum_norm.write_inverse_operator(cfg.fname.inverse_operator_surface(subject=subject), inv_sur, overwrite=True)

print('>> Apply inverse operators (surface) on evoked (distractor)')
stc_evoked_distractor_sur = mne.minimum_norm.apply_inverse(evoked_distractor, inv_sur, method=cfg.method_sur, lambda2=lambda2_sur, verbose=cfg.mne_log_level)
stc_evoked_distractor_sur.save(cfg.fname.source_estimate_evoked_distractor_surface(subject=subject), overwrite=True)

print('>> Apply inverse operators (surface) on evoked (target)')
stc_evoked_target_sur = mne.minimum_norm.apply_inverse(evoked_target, inv_sur, method=cfg.method_sur, lambda2=lambda2_sur, verbose=cfg.mne_log_level)
stc_evoked_target_sur.save(cfg.fname.source_estimate_evoked_target_surface(subject=subject), overwrite=True)

print('>> Apply inverse operators (surface) on evoked (difference)')
stc_evoked_diff_sur = mne.minimum_norm.apply_inverse(evoked_diff, inv_sur, method=cfg.method_sur, lambda2=lambda2_sur, verbose=cfg.mne_log_level)
stc_evoked_diff_sur.save(cfg.fname.source_estimate_evoked_difference_surface(subject=subject), overwrite=True)

print('>> Apply inverse operators (surface) on evoked (total)')
stc_evoked_sur = mne.minimum_norm.apply_inverse(evoked_total, inv_sur, method=cfg.method_sur, lambda2=lambda2_sur, verbose=cfg.mne_log_level)
stc_evoked_sur.save(cfg.fname.source_estimate_evoked_total_surface(subject=subject), overwrite=True)

# ---------------------------------------------------------------------------------------------------------
# Volume
lambda2_vol = 1.0 / cfg.snr_vol ** 2

print('>> Read volume source space')
# The file is generated by setup_volume_source_space.py
vol_src = mne.read_source_spaces(cfg.fname.volume_source_space)

print('>> Make forward solution (volume)')
fwd_vol = mne.make_forward_solution(
    epochs.info, trans=trans, src=vol_src, bem=fsaverage_bem, eeg=True, mindist=5.0, n_jobs=8)
mne.write_forward_solution(cfg.fname.forward_solution_volume(subject=subject), fwd_vol, overwrite=True)

print('>> Compute inverse operator (volume)')
inv_vol = mne.minimum_norm.make_inverse_operator(epochs.info, fwd_vol, noise_cov, verbose=cfg.mne_log_level, loose=1, depth=0.8)

# Requires nibabel, nilearn
print('>> Apply inverse operators (volume)')
stc_vol = mne.minimum_norm.apply_inverse(evoked_total, inv_vol, lambda2_vol, cfg.method_vol, verbose=cfg.mne_log_level)
stc_vol.save(cfg.fname.source_estimate_evoked_total_volume(subject=subject), overwrite=True)


