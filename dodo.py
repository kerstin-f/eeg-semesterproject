"""
Do-it script to execute the entire pipeline using the doit tool:
http://pydoit.org

All the filenames are defined in config.py
"""
import config as cfg
import utils

subjects = utils.generate_subjects_list(cfg.subjects)
subjects_manual = utils.generate_subjects_list(cfg.subjects_manual)
print('The following subjects will be processed: ' + str(subjects))
print('The following subjects will be processed manually: ' + str(subjects_manual))

# Configuration for the "doit" tool.
DOIT_CONFIG = dict(
    # While running scripts, output everything the script is printing to the
    # screen.
    verbosity=2,
    failure_verbosity=2,
    # When the user executes "doit list", list the tasks in the order they are
    # defined in this file, instead of alphabetically.
    sort='definition',
)

def task_check_system():
    """Check the system dependencies."""
    return dict(
        file_dep=['check_system.py'],
        targets=[cfg.fname.system_check],
        actions=['python check_system.py'],
    )

# This example task executes a single analysis script for each subject, giving
# the subject as a command line parameter to the script.
def task_apply_filter():
    """Step 00: Perform band-pass filtering on raw data."""
    # Run the example script for each subject in a sub-task.
    for subject in subjects:
        yield dict(
            # This task should come after `task_check`
            task_dep=['check_system'],

            # A name for the sub-task: set to the name of the subject
            name=subject,

            # If any of these files change, the script needs to be re-run. Make
            # sure that the script itself is part of this list!
            file_dep=['00_apply_filter.py'],

            # The files produced by the script
            targets=[cfg.fname.filtered(subject=subject)],

            # How the script needs to be called. Here we indicate it should
            # have one command line parameter: the name of the subject.
            actions=['python 00_apply_filter.py %s' % subject],
        )

def task_prepare_raw():
    """Step 01: Prepare raw for subsequent analysis."""
    for subject in subjects:
        yield dict(
            task_dep=['apply_filter'],
            name=subject,
            file_dep=['00_apply_filter.py', '01_prepare_raw.py'],
            targets=[cfg.fname.prepared(subject=subject)],
            actions=['python 01_prepare_raw.py %s' % subject],
            verbosity=2,
        )

def task_run_ica():
    """Step 02: Run ICA on raw data filtered with 1 Hz highpass."""
    # Run ICA only for subjects to be processed manually
    for subject in subjects_manual:
        yield dict(
            task_dep=['prepare_raw'],
            name=subject,
            file_dep=['00_apply_filter.py', '01_prepare_raw.py', '02_run_ica.py'],
            targets=[cfg.fname.ica_result_manual(subject=subject), cfg.fname.badComponents_eog(subject=subject)],
            actions=['python 02_run_ica.py %s' % subject],
        )

def task_apply_ica():
    """Step 03: Remove effects of ICs from raw that are marked for exclusion."""
    for subject in subjects:
        yield dict(
            task_dep=['run_ica'],
            name=subject,
            file_dep=['00_apply_filter.py', '01_prepare_raw.py', '02_run_ica.py', '03_apply_ica.py'],
            targets=[cfg.fname.raw_cleaned(subject=subject)],
            actions=['python 03_apply_ica.py %s' % subject],
        )

def task_make_epochs():
    """Step 04: Create epochs by segementing data according to event codes."""
    for subject in subjects:
        yield dict(
            task_dep=['apply_ica'],
            name=subject,
            file_dep=['00_apply_filter.py', '01_prepare_raw.py', '02_run_ica.py', '03_apply_ica.py', '04_make_epochs.py'],
            targets=[cfg.fname.epoched(subject=subject)],
            actions=['python 04_make_epochs.py %s' % subject],
        )

def task_make_evokeds():
    """Step 05: Create evoked datasets by averaging over different subsets of epochs."""
    for subject in subjects:
        yield dict(
            task_dep=['make_epochs'],
            name=subject,
            file_dep=['00_apply_filter.py', '01_prepare_raw.py', '02_run_ica.py', '03_apply_ica.py', '04_make_epochs.py','05_make_evokeds.py'],
            targets=[cfg.fname.evokeds(subject=subject)],
            actions=['python 05_make_evokeds.py %s' % subject],
        )

def task_analyze_single():
    """Step 06: Analyze ERP peak on single-subject level."""
    for subject in subjects:
        yield dict(
            task_dep=['make_evokeds'],
            name=subject,
            file_dep=['00_apply_filter.py', '01_prepare_raw.py', '02_run_ica.py', '03_apply_ica.py', '04_make_epochs.py', '05_make_evokeds.py', '06_analyze_single.py'],
            targets=[],
            actions=['python 06_analyze_single.py %s' % subject],
        )

# def task_analyze_all():
#     """Step 07: Perform group analysis of ERP peak."""
#     if len(subjects) > 1: 
#         yield dict(
#             task_dep=['make_evokeds'],
#             name='all subjects',
#             file_dep=['00_apply_filter.py', '01_prepare_raw.py', '02_run_ica.py', '03_apply_ica.py', '04_make_epochs.py', '05_make_evokeds.py', '07_analyze_all.py'],
#             targets=[cfg.fname.grand_ave],
#             actions=['python 07_analyze_all.py'],
#         )
#     else:
#         print('No Grand Average is generated because only one subject is processed.')


# def task_run_decoding():
#     """Step 08: Perform decoding analysis."""
#     for subject in subjects:
#         yield dict(
#             task_dep=['make_evokeds'],
#             name=subject,
#             file_dep=['00_apply_filter.py', '01_prepare_raw.py', '03_apply_ica.py', '04_make_epochs.py', '05_make_evokeds.py', '08_run_decoding.py'],
#             targets=[],
#             actions=['python 08_run_decoding.py %s' % subject],
#         )

# def task_source_localization():
#     """Step 09: Perform source localization."""
#     for subject in subjects:
#         yield dict(
#             task_dep=['make_evokeds'],
#             name=subject,
#             file_dep=['00_apply_filter.py', '01_prepare_raw.py', '03_apply_ica.py', '04_make_epochs.py','05_make_evokeds.py', '09_source_localization.py'],
#             targets=[
#                 cfg.fname.noise_covariance(subject=subject), 
#                 cfg.fname.forward_solution_surface(subject=subject), 
#                 cfg.fname.forward_solution_volume(subject=subject), 
#                 cfg.fname.source_estimate_evoked_total_surface(subject=subject),
#                 cfg.fname.source_estimate_evoked_total_volume(subject=subject),
#                 cfg.fname.source_estimate_evoked_target_surface(subject=subject),
#                 cfg.fname.source_estimate_evoked_distractor_surface(subject=subject),
#                 cfg.fname.source_estimate_evoked_difference_surface(subject=subject)
#                 ],
#             actions=['python 09_source_localization.py %s' % subject],
#         )

# def task_source_localization_report():
#     """Step 10: Create report for source localization.."""
#     for subject in subjects:
#         yield dict(
#             task_dep=['source_localization'],
#             name=subject,
#             file_dep=['00_apply_filter.py', '01_prepare_raw.py', '03_apply_ica.py', '04_make_epochs.py','05_make_evokeds.py', '09_source_localization.py', '10_source_localization_report.py'],
#             targets=[],
#             actions=['python 10_source_localization_report.py %s' % subject],
#         )


