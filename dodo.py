"""
Do-it script to execute the entire pipeline using the doit tool:
http://pydoit.org

All the filenames are defined in config.py
"""
from config import fname, subjects
import os

# Configuration for the "doit" tool.
DOIT_CONFIG = dict(
    # While running scripts, output everything the script is printing to the
    # screen.
    verbosity=2,

    # When the user executes "doit list", list the tasks in the order they are
    # defined in this file, instead of alphabetically.
    sort='definition',
)

def task_delete_reports():
    """Delete all files in the report directory. Use with caution!"""
    return dict(
        file_dep=['00_filtering.py', '01_make_epochs.py', '02_apply_ica.py', '03_make_evoked.py', '05_decoding.py'],
        actions=['python delete_reports.py']
    )

def task_check():
    """Check the system dependencies."""
    return dict(
        file_dep=['check_system.py'],
        targets=[fname.system_check],
        actions=['python check_system.py']
    )

# This example task executes a single analysis script for each subject, giving
# the subject as a command line parameter to the script.
def task_filtering():
    """Step 00: An example analysis step that is executed for each subject."""
    # Run the example script for each subject in a sub-task.
    for subject in subjects:
        yield dict(
            # This task should come after `task_check`
            task_dep=['check'],

            # A name for the sub-task: set to the name of the subject
            name=subject,

            # If any of these files change, the script needs to be re-run. Make
            # sure that the script itself is part of this list!
            file_dep=['00_filtering.py'],

            # The files produced by the script
            targets=[fname.filtering(subject=subject)],

            # How the script needs to be called. Here we indicate it should
            # have one command line parameter: the name of the subject.
            actions=['python 00_filtering.py %s' % subject],
        )

def task_make_epochs():
    """Step 01: An example analysis step that is executed for each subject."""
    # Extract epochs for each subject.
    for subject in subjects:
        yield dict(
            task_dep=['filtering'],
            name=subject,
            file_dep=['00_filtering.py', '01_make_epochs.py'],
            targets=[fname.epoching(subject=subject)],
            actions=['python 01_make_epochs.py %s' % subject],
        )

def task_apply_ica():
    """Step 02: An example analysis step that is executed for each subject."""
    # Extract epochs for each subject.
    for subject in subjects:
        yield dict(
            task_dep=['make_epochs'],
            name=subject,
            file_dep=['00_filtering.py', '01_make_epochs.py', '02_apply_ica.py'],
            targets=[fname.cleaned_epochs(subject=subject)],
            actions=['python 02_apply_ica.py %s' % subject],
        )

def task_make_evoked():
    """Step 03: An example analysis step that is executed for each subject."""
    # Extract epochs for each subject.
    # Problem: nur einmal ausführen??
    for subject in subjects:
        yield dict(
            task_dep=['apply_ica'],
            name=subject,
            file_dep=['00_filtering.py', '01_make_epochs.py', '02_apply_ica.py', '03_make_evoked.py'],
            targets=[fname.evokeds(subject=subject)],
            actions=['python 03_make_evoked.py %s' % subject],
        )

def task_decoding():
    """Step 03: An example analysis step that is executed for each subject."""
    # Extract epochs for each subject.
    # Problem: nur einmal ausführen??
    for subject in subjects:
        yield dict(
            task_dep=['make_evoked'],
            name=subject,
            file_dep=['00_filtering.py', '01_make_epochs.py', '02_apply_ica.py', '03_make_evoked.py', '05_decoding.py'],
            targets=[],
            actions=['python 05_decoding.py %s' % subject],
        )
