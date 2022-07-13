""""
Entire epochs are fed into a logistic regression model. Decoding performance
then tells how well the classifier could predict which epoch belongs to
which condition.
"""

import argparse
import mne
import numpy as np
import pandas as pd
import mne

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from mne.decoding import Scaler, Vectorizer
from matplotlib import pyplot as plt
from config import fname, analyze_channels, decoding_n_splits, random_state

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub-###', help='The subject to process')
parser.add_argument('condition_1')
parser.add_argument('condition_2')
args = parser.parse_args()
subject = args.subject
condition_1 = args.condition_1
condition_2 = args.condition_2

print('Processing subject:', subject)
print('Contrasting conditions:{condition1} – {condition2}')

# The evoked data sets are created by averaging different conditions.
epochs = mne.read_epochs(fname.epochs_cleaned(subject=subject), preload=True)

# We special-case the average reference here to work around a situation
# where e.g. `analyze_channels` might contain only a single channel:
# `concatenate_epochs` below will then fail when trying to create /
# apply the projection. We can avoid this by removing an existing    
# average reference projection here, and applying the average reference    
# directly – without going through a projector.

epochs.set_eeg_reference('average')
epochs.pick(analyze_channels)

epochs_conds = [condition_1, condition_2]

# Problem: 01_make_epoching.py does not differentiates different conditions
epochs = mne.concatenate_epochs([epochs[epochs_conds[0]],epochs[epochs_conds[1]]])

n_cond1 = len(epochs[epochs_conds[0]])
n_cond2 = len(epochs[epochs_conds[1]])

X = epochs.get_data()
y = np.r_[np.ones(n_cond1), np.zeros(n_cond2)]

classification_pipeline = make_pipeline(
    Scaler(scalings='mean'),
    Vectorizer(),
+   LogisticRegression(
        solver='liblinear',  # much faster than the default
        random_state=random_state,
        n_jobs=1,
    )
)

# Run the classification, and evaluate it via a cross-validation procedure.
cv = StratifiedKFold(
    shuffle=True,
    random_state=random_state,
    n_splits=decoding_n_splits,
)
scores = cross_val_score(
    estimator=classification_pipeline,
    X=X,
    y=y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=1
)

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)