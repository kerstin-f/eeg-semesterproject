
import os
import mne
import numpy as np
import pandas as pd
from scipy import linalg
from config import fname, bids_root
from mne_bids.read import _from_tsv,_drop
from mne_bids import (BIDSPath, read_raw_bids)

def import_data(subject, task):
    bids_path = BIDSPath(subject=subject,
                     task=task,
                     session=task,
                     datatype='eeg', 
                     suffix='eeg',
                     root=bids_root)
    
    # read_raw_bids automatically
    raw = read_raw_bids(bids_path=bids_path)
    # Fix the annotations reading
    _read_annotations_core(bids_path,raw)

    # Missing: _crop_data(cfg, raw, subject): Crop the data to the desired duration.

    raw.load_data()

    # Add channel locations
    raw.drop_channels(['HEOG_left', 'HEOG_right', 'VEOG_lower'])
    raw.set_montage('standard_1020',match_case=False)

    # Missing: _drop_channels_func(cfg, raw, subject, session): Drop channels from the data.
    # Missing: _fix_stim_artifact_func(cfg: SimpleNamespace,raw: mne.io.BaseRaw): Fix stimulation artifact in the data.
    # Not required (?): _find_bad_channels(cfg=cfg, raw=raw, subject=subject, session=session,task=get_task(), run=run): Find and mark bad MEG channels.

    return raw


def _read_annotations_core(bids_path,raw):
    tsv=os.path.join(bids_path.directory,bids_path.update(suffix="events",extension=".tsv").basename)
    _handle_events_reading_core(tsv,raw)

def _handle_events_reading_core(events_fname, raw):
    """Read associated events.tsv and populate raw.
    Handle onset, duration, and description of each event.
    """
    events_dict = _from_tsv(events_fname)

    if ('value' in events_dict) and ('trial_type' in events_dict):
        events_dict = _drop(events_dict, 'n/a', 'trial_type')
        events_dict = _drop(events_dict, 'n/a', 'value')

        descriptions = np.asarray([a+':'+b for a,b in zip(events_dict["trial_type"],events_dict["value"])], dtype=str)  
        
    # Get the descriptions of the events
    elif 'trial_type' in events_dict:
          
        # Drop events unrelated to a trial type
        events_dict = _drop(events_dict, 'n/a', 'trial_type')
        descriptions = np.asarray(events_dict['trial_type'], dtype=str)
          
    # If we don't have a proper description of the events, perhaps we have
    # at least an event value?
    elif 'value' in events_dict:
        # Drop events unrelated to value
        events_dict = _drop(events_dict, 'n/a', 'value')
        descriptions = np.asarray(events_dict['value'], dtype=str)
    # Worst case, we go with 'n/a' for all events
    else:
        descriptions = 'n/a'
    # Deal with "n/a" strings before converting to float
    ons = [np.nan if on == 'n/a' else on for on in events_dict['onset']]
    dus = [0 if du == 'n/a' else du for du in events_dict['duration']]
    onsets = np.asarray(ons, dtype=float)
    durations = np.asarray(dus, dtype=float)
    # Keep only events where onset is known
    good_events_idx = ~np.isnan(onsets)
    onsets = onsets[good_events_idx]
    durations = durations[good_events_idx]
    descriptions = descriptions[good_events_idx]
    del good_events_idx
    # Add Events to raw as annotations
    annot_from_events = mne.Annotations(onset=onsets,
                                        duration=durations,
                                        description=descriptions,
                                        orig_time=None)
    raw.set_annotations(annot_from_events)
    return raw

def load_precomputed_ica(subject):
    # returns ICA and badComponents (starting at component = 0).
    # Note the existance of add_ica_info in case you want to plot something.

    # import the eeglab ICA. I used eeglab because the "amica" ICA is a bit more powerful than runica
    # ica = mne.preprocessing.read_ica_eeglab(fn +'.set')

    # I have to use this, because the montage is first set before it is subsetted. Thereby it requires chaninfo for all channels, not only the channels used i ICA...
    ica = sp_read_ica_eeglab(fname.precomputed_ica_set(subject=subject))
    # Potentially for plotting one might want to copy over the raw.info, but in this function we dont have access / dont want to load it
    # ica.info = raw.info
    ica._update_ica_names()
    badComps = np.loadtxt(fname.precomputed_ica_tsv(subject=subject),delimiter="\t")
    badComps -= 1 # start counting at 0
    
    # if only a single component is in the file, we get an error here because it is an ndarray with n-dim = 0.
    if len(badComps.shape) == 0:
        badComps = [float(badComps)]
    return ica, badComps

def add_ica_info(raw, ica):
    # This function exists due to a MNE bug: https://github.com/mne-tools/mne-python/issues/8581
    # In case you want to plot your ICA components, this function will generate a ica.info
    ch_raw = raw.info['ch_names']
    ch_ica = ica.ch_names

    ix = [k for k,c in enumerate(ch_raw) if c in ch_ica and not c in raw.info['bads']]
    info = raw.info.copy()
    mne.io.pick.pick_info(info, ix, copy=False)
    ica.info = info
    return ica

def load_precomputed_badData(subject):
    # return precomputed annotations and bad channels (first channel = 0)
    tmp = pd.read_csv(fname.precomputed_badSegments(subject=subject))
    annotations = mne.Annotations(tmp.onset, tmp.duration, tmp.description)
    # Unfortunately MNE assumes that csv files are in milliseconds and only txt files in seconds.. wth?
    badChannels = np.loadtxt(fname.precomputed_badChannels(subject=subject), delimiter='\t')
    badChannels = badChannels.astype(int)
    badChannels -= 1 # start counting at 0
    return annotations, badChannels


def sp_read_ica_eeglab(fname, *, verbose=None):
    """Load ICA information saved in an EEGLAB .set file.

    Parameters
    ----------
    fname : str
        Complete path to a .set EEGLAB file that contains an ICA object.
    %(verbose)s

    Returns
    -------
    ica : instance of ICA
        An ICA object based on the information contained in the input file.
    """
    from scipy import linalg
    eeg = mne.preprocessing.ica._check_load_mat(fname, None)
    info, eeg_montage, _ = mne.preprocessing.ica._get_info(eeg)
    mne.pick_info(info, np.round(eeg['icachansind']).astype(int) - 1, copy=False)
    info.set_montage(eeg_montage)
    

    rank = eeg.icasphere.shape[0]
    n_components = eeg.icaweights.shape[0]

    ica = mne.preprocessing.ica.ICA(method='imported_eeglab', n_components=n_components)

    ica.current_fit = "eeglab"
    ica.ch_names = info["ch_names"]
    ica.n_pca_components = None
    ica.n_components_ = n_components

    n_ch = len(ica.ch_names)
    assert len(eeg.icachansind) == n_ch

    ica.pre_whitener_ = np.ones((n_ch, 1))
    ica.pca_mean_ = np.zeros(n_ch)

    assert eeg.icasphere.shape[1] == n_ch
    assert eeg.icaweights.shape == (n_components, rank)

    # When PCA reduction is used in EEGLAB, runica returns
    # weights= weights*sphere*eigenvectors(:,1:ncomps)';
    # sphere = eye(urchans). When PCA reduction is not used, we have:
    #
    #     eeg.icawinv == pinv(eeg.icaweights @ eeg.icasphere)
    #
    # So in either case, we can use SVD to get our square whitened
    # weights matrix (u * s) and our PCA vectors (v) back:
    use = eeg.icaweights @ eeg.icasphere
    use_check = linalg.pinv(eeg.icawinv)
    if not np.allclose(use, use_check, rtol=1e-6):
        warn('Mismatch between icawinv and icaweights @ icasphere from EEGLAB '
             'possibly due to ICA component removal, assuming icawinv is '
             'correct')
        use = use_check
    u, s, v = mne.preprocessing.ica._safe_svd(use, full_matrices=False)
    ica.unmixing_matrix_ = u * s
    ica.pca_components_ = v
    ica.pca_explained_variance_ = s * s
    ica.info = info
    ica._update_mixing_matrix()
    ica._update_ica_names()
    return ica