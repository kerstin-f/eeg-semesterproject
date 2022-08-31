import os
import mne
import numpy as np
import pandas as pd
import config as cfg
from typing import List, Dict, Tuple
from logging import warn
from mne_bids.read import _from_tsv,_drop
from mne_bids import (BIDSPath, read_raw_bids)
from scipy.stats.mstats import winsorize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from lxml import etree
from lxml import html
import seaborn as sns

def import_raw(subject:str, task:str) -> mne.io.Raw:
    bids_root = cfg.raw_data_dir + '/' + cfg.task + '/'
    bids_path = BIDSPath(subject=subject,
                     task=task,
                     session=task,
                     datatype='eeg', 
                     suffix='eeg',
                     root=bids_root)
    
    # read_raw_bids automatically
    raw = read_raw_bids(bids_path=bids_path, verbose=cfg.mne_log_level)
    
    # Fix the annotations reading
    _handle_events_reading_core(subject,raw)

    raw.load_data()

    # Missing: _drop_channels_func(cfg, raw, subject, session): Drop channels from the data.
    # Missing: _fix_stim_artifact_func(cfg: SimpleNamespace,raw: mne.io.BaseRaw): Fix stimulation artifact in the data.

    return raw

def read_event_keys_from_json() -> Dict:
    df = pd.read_json(cfg.fname.event_code_values(task=cfg.task), 'Levels')
    evts_codes = dict(df['value'][0])
    evts_desc = {}
    # Key: events, Value: description
    # (e.g. 'stimulus:11': 'block target A, trial stimulus A')
    for v in evts_codes.items():
        desc = v[1].split(' - ')
        evts_desc[str.lower(desc[0]) + ':' + v[0]] = desc[1]
    return evts_desc


def _handle_events_reading_core(subject:str, raw:mne.io.Raw):
    """Read associated events.tsv and populate raw.
    Handle onset, duration, and description of each event.
    """
    events_dict = _from_tsv(cfg.fname.events(subject=subject, task=cfg.task))

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

def load_ica(subject:str, manual:bool) -> mne.preprocessing.ICA:
    if manual:
        ica = mne.preprocessing.read_ica(fname=cfg.fname.ica_result_manual(subject=subject))
    else:
        # I have to use this, because the montage is first set before it is subsetted. Thereby it requires chaninfo for all channels, not only the channels used i ICA...
        ica = sp_read_ica_eeglab(cfg.fname.ica_result_precomputed(subject=subject, task=cfg.task))
        # Potentially for plotting one might want to copy over the raw.info, but in this function we dont have access / dont want to load it
        # ica.info = raw.info
        ica._update_ica_names()
    return ica

def process_file_content(arr: np.ndarray, zero_based: bool) -> List[int]:
    arr = arr.astype(int)
    if not zero_based:
        arr -= 1 # start counting at 0
    if len(arr.shape) == 0:
        return [int(arr)]
    else:
        return arr.tolist()

def load_badComponents(subject:str, manual:bool) -> List[int]:
    badComponents = []
    if manual and os.path.isfile(cfg.fname.badComponents_manual(subject=subject)):
        content = np.loadtxt(cfg.fname.badComponents_manual(subject=subject),delimiter="\t")
        badComponents = process_file_content(content, zero_based=True)
    elif manual:
        print('>>> A file with manually identifed bad ICA components could not be found. Precomputed bad ICA components will be used instead.')
        if os.stat(cfg.fname.badComponents_eog(subject=subject)).st_size:
            content = np.loadtxt(cfg.fname.badComponents_eog(subject=subject),delimiter="\t")
            badComponents = process_file_content(content, zero_based=True)
    else:
        if os.stat(cfg.fname.badComponents_precomputed(subject=subject, task=cfg.task)).st_size:
            content = np.loadtxt(cfg.fname.badComponents_precomputed(subject=subject, task=cfg.task),delimiter="\t")
            badComponents = process_file_content(content, zero_based=False)
    return badComponents

def add_ica_info(raw:mne.io.Raw, ica:mne.preprocessing.ICA) -> mne.preprocessing.ICA:
    # This function exists due to a MNE bug: https://github.com/mne-tools/mne-python/issues/8581
    # In case you want to plot your ICA components, this function will generate a ica.info
    ch_raw = raw.info['ch_names']
    ch_ica = ica.ch_names

    ix = [k for k,c in enumerate(ch_raw) if c in ch_ica and not c in raw.info['bads']]
    info = raw.info.copy()
    mne.io.pick.pick_info(info, ix, copy=False)
    ica.info = info
    return ica

def load_badSegments(subject:str, manual:bool) -> mne.Annotations:
    # return precomputed annotations and bad channels (first channel = 0)
    if manual and os.path.isfile(cfg.fname.badSegments_manual(subject=subject)):
        fname = cfg.fname.badSegments_manual(subject=subject)
    else:
        if manual:
            print('>>> A file with manually annotated bad segments could not be found. Precomputed bad segments will be used instead.')
        fname = cfg.fname.badSegments_precomputed(subject=subject, task=cfg.task)
    badSegments = pd.read_csv(fname)
    return mne.Annotations(badSegments.onset, badSegments.duration, badSegments.description)

def load_badChannels(subject:str, manual:bool) -> List[int]:
    badChannels = []
    if manual and os.path.isfile(cfg.fname.badChannels_manual(subject=subject)):
        fname = cfg.fname.badChannels_manual(subject=subject)
    else:
        if manual:
            print('>>> A file with manually annotated bad channels could not be found. Precomputed bad channels will be used instead.')
        fname = cfg.fname.badChannels_precomputed(subject=subject, task=cfg.task)
    
    if os.stat(fname).st_size:
        content = np.loadtxt(fname,delimiter='\t')
        badChannels = process_file_content(content,zero_based=False)

    return badChannels
    
def sp_read_ica_eeglab(fname:str, *, verbose=None) -> mne.preprocessing.ICA:
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


# ---------------------------------------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------------------------------------

def generate_subjects_list(subjects:any) -> List[str]:

    bids_root = cfg.raw_data_dir + "/" + cfg.task 

    # All subjects from 1 to N
    count = 0
    for _, dirs, _ in os.walk(bids_root):
        for dir in dirs:
            if(dir.startswith('sub-')):
                count += 1
    all_subjects = ["%.3d" % i for i in range(1,count+1)]
    msg = 'Subject(s) {} cannot be found'.format(subjects)
    if subjects == 'all': 
        return all_subjects
    elif isinstance(subjects, str):
        assert subjects in all_subjects, msg
        return [subjects]
    elif isinstance(subjects, list):
        assert all ([s in all_subjects for s in subjects]), msg
        return subjects
    elif isinstance(subjects, int):
        assert len(all_subjects) > subjects, msg
        return all_subjects[:subjects]
    else:
        raise ValueError('Wrong setting for subjects. The dataset contains {} subjects.'.format(len(all_subjects)))

# Callback for averaging epochs
def mean(d):
    return np.mean(d,axis=0)
def winsor(d):
    return np.mean(winsorize(d,axis=0,limits=(0.2,0.2)),axis=0)
def median(d):
    return np.median(d,axis=0)


def compute_regression_baselining(epochs: mne.Epochs, ch: str) -> Dict:
    # Problem: Wieso einmal alle Channels und bei baseline_predictor nur ein Channel??
    # Problem: Relevant, ob gleiche oder andere Anzahl an Distractor und Target Events??
    condition_predictors = []
    for condition in cfg.conditions:
        condition_predictors.append(epochs.events[:,2] == epochs.event_id[condition])
    baseline_predictor = (epochs.copy()
                            .crop(*cfg.baseline)
                            .pick_channels([ch])
                            .get_data()     # convert to NumPy array
                            .mean(axis=-1)  # average across timepoints
                            .squeeze())      # only 1 channel, so remove singleton dimension

    baseline_predictor *= 1e6  # convert V → μV
    design_matrix = np.vstack(condition_predictors +
                               [baseline_predictor,
                               baseline_predictor * condition_predictors[0]]).T

    regression_model = mne.stats.linear_regression(epochs,
                                            design_matrix,
                                            names=cfg.conditions + ['Baseline','Interaction:Baseline-'+cfg.conditions[0]])
    return regression_model


# ---------------------------------------------------------------------------------------------------------
# ERP PEAK ANALYSIS
# ---------------------------------------------------------------------------------------------------------

def get_peak_info(evoked: mne.Evoked) -> Tuple[str,Figure]:
    measures = evoked.get_peak(tmin=cfg.meas_tmin, tmax=cfg.meas_tmax, mode='pos', return_amplitude=True)

    # Create dataframe to plot measures
    df = pd.DataFrame({'Channel': measures[0],'Peak Latency [ms]': measures[1],'Peak Amplitude [µV]': measures[2]}, index=[0])
    df['Peak Latency [ms]'] = df['Peak Latency [ms]'].astype(float)
    df['Peak Amplitude [µV]'] = df['Peak Amplitude [µV]'].astype(float)
    df['Peak Latency [ms]'] *= 1e3
    df['Peak Amplitude [µV]'] *= 1e6
    html_df = df.to_html(index=False, col_space='150px', justify='left')
    html_df = '<center>' + html_df + '</center>'

    # Create plot of marked peak in difference wave
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    evoked.plot(axes=ax, time_unit='ms', show=False, titles='P3 peak in difference wave')
    ax.plot(df['Peak Latency [ms]'], df['Peak Amplitude [µV]'], marker='*', color='C6')
    ax.axvspan(*(np.array([cfg.meas_tmin, cfg.meas_tmax])* 1e3), facecolor='C1', alpha=0.3)
    #ax.set_xlim(-50, 150)  # Show zoomed in around peak
    return html_df, fig

def perform_t_test(t_test_data:np.ndarray, info_obj, picks:np.ndarray):
    T0, p_values, H0 = mne.stats.permutation_t_test(t_test_data, cfg.n_permutations_t, tail=1)
    # Create Evoked object using p-values
    evoked_p = mne.EvokedArray(-np.log10(p_values)[:, np.newaxis], info_obj.info, tmin=0.0)
    mask = p_values[:, np.newaxis] <= 0.05
    # Restrict topomap plot to significant channels
    sig_sensors = picks[p_values <= 0.05]
    sig_sensor_names = [info_obj.ch_names[k] for k in sig_sensors]
    evoked_p.pick_channels(sig_sensor_names)
    # Explicitly compute split for visualization
    # Compute permutation-corrected threshold for t-values
    thresh = round(cfg.n_permutations_t * 0.05)
    split = np.argpartition(H0, -thresh)[-thresh:]

    plt.figure()
    fig_evoked_topomap = evoked_p.plot_topomap(times=[0], scalings=1,
                                               time_format='', cmap='Reds', vmin=0., vmax=np.max,
                                               units='-log10(p)', cbar_fmt='-%0.1f', mask=mask,
                                               size=2, show_names=True)

    # Adjust channel names font size and position
    def myfunc(x):
        return hasattr(x, 'set_fontsize')
    for tt in fig_evoked_topomap.findobj(myfunc):
        if tt.get_text() in evoked_p.ch_names:
            p = tt.get_position()
            tt.set_position((p[0], p[1]+0.01))
            tt.set_fontsize(10)

    plt.figure()
    plt.autoscale()
    fig_histogram, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), layout='tight')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax = sns.histplot(data=[H0[split],H0], legend=False, bins=70, alpha=1.0)
    ax.margins(0)
    ax.set_xlim([0, 5])
    ax.set_xlabel('t-value')
    ax.legend(['t-value','Top 5%'])

    return fig_evoked_topomap, fig_histogram

def perform_permutation_cluster_test(permutation_test_data:np.ndarray, diff_choi) -> Figure:
    F_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(permutation_test_data, 
                                                                        n_permutations=cfg.n_permutations_cluster,
                                                                        threshold=cfg.thresh_cluster,
                                                                        stat_fun=mne.stats.f_oneway,
                                                                        adjacency=None,
                                                                        tail=1,
                                                                        out_type='mask')
    times = diff_choi.times
    plt.figure()
    fig_permutation_cluster_test, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), layout='tight')
    ax1.plot(times, diff_choi.get_data()[0,:], label='Difference', color=cfg.color_dict['Difference'])
    ax1.set_title('Channel : ' + cfg.choi)
    ax1.set_ylabel('Volt [µV]')
    ax1.legend()

    for ix, cluster in enumerate(clusters):
        c = cluster[0]
        if cluster_pv[ix] <= 0.05:
            span = ax2.axvspan(times[c.start], times[c.stop - 1], color='r', alpha=0.3)
            ax2.legend((span, ), ('Cluster p-value <= 0.05', ))
        else:
            span = ax2.axvspan(times[c.start], times[c.stop - 1], color='g', alpha=0.3)
    plt.plot(times, F_obs, color='g')
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("F-values")
    return fig_permutation_cluster_test


# ---------------------------------------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------------------------------------
def figure_to_image(figure):
    canvas = FigureCanvas(figure)
    ax = figure.gca()
    ax.margins(0)
    canvas.draw()  
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(figure.canvas.get_width_height()[::-1] + (3,))
   
    return image

def combine_figures(figure1, title_figure1, figure2, title_figure2, figure3 = None, title_figure3 = None, fig_size=(16, 7)):

    if isinstance(figure1, list):
        figure1 = figure1[0]
    if not isinstance(figure1, plt.Figure):
        raise Exception('Figure 1 is not a matplotlib.Figure')

    if isinstance(figure2, list):
        figure2 = figure2[0]
    if not isinstance(figure2, plt.Figure):
        raise Exception('Figure 2 is not a matplotlib.Figure')

    if figure3:
        if isinstance(figure3, list):
            figure3 = figure3[0]
        if not isinstance(figure3, plt.Figure):
            raise Exception('Figure 3 is not a matplotlib.Figure')

    num_figs = 2
    #figsize=(16, 7)
    if figure3:
        num_figs = 3
        #figsize=(20, 7)

    pad_title = 30
    plt.figure()
    fig_sidebyside, ax_arr = plt.subplots(1, num_figs, figsize=fig_size, layout='tight')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    ax = ax_arr[0]
    if title_figure1:
        ax.set_title(title_figure1, pad=pad_title)
    ax.set_axis_off()
    ax.set_anchor('N')
    ax.margins(0)
    ax.imshow(figure_to_image(figure1))

    ax = ax_arr[1]
    if title_figure2:
        ax.set_title(title_figure2, pad=pad_title)
    ax.set_axis_off()
    ax.set_anchor('N')
    ax.margins(0)
    ax.imshow(figure_to_image(figure2))

    if figure3:
        ax = ax_arr[2]
        if title_figure3:
            ax.set_title(title_figure3, pad=pad_title)
        ax.set_axis_off()
        ax.set_anchor('N')
        ax.margins(0)
        ax.imshow(figure_to_image(figure3))

    return fig_sidebyside

tree = None
def add_html(report, element, title, tags):
    if cfg.add_html_snippets:
        split_el = element.split('.')
        if not isinstance(element,str) or not len(split_el) == 2:
            raise Exception('util.add_html: element "{}" is not a valid identifier.'.format(element))
        # Load once
        global tree
        fname = 'report_snippets.html'
        if tree is None:
            with open(fname, "r") as f:
                tree = html.parse(f)

        try:
            xpath = '//{}/{}'.format(split_el[0].lower(),split_el[1].lower())
            node = tree.xpath(xpath)
            html_str = etree.tostring(node[0]).decode("utf-8")
        except:
            raise Exception('util.add_html: xpath "{}" not found in file "{}.'.format(xpath, fname))
        report.add_html(html_str, title, tags=tags)
