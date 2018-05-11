import numpy as np

import csv
import os
import json
import pickle

from audio.signals import mel_spectrogram

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

import logging
from scipy.io import wavfile
import timeit


csvfile = open('ESC-50-master/meta/esc50.csv' ,'rb')
lines = csv.reader(csvfile)

data = []
label = []

for line in list(lines)[1:]:
    file_name = line[0]
    file_path = os.path.join('ESC-50-master/audio', file_name)
    cls = int(line[2])

    
    if os.path.exists(file_path) and cls in [1,2,20]:
        sr, y = wavfile.read(file_path)
        mel_spec_power = mel_spectrogram(y, sr)
        data.append(mel_spec_power)
        label.append(cls)

X = np.array(data)
y = np.array(label, dtype = np.int)

X = X.reshape(len(X), -1)

def train(X, y):
    """
    Train Random Forest

    :return: pipeline, best_param, best_estimator, perf
    """

    logging.info('Splitting train and test set. Test set size: 0.25%')

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=0,
                                                        stratify=y)

    logging.info('Train set size: {0}. Test set size: {1}'.format(y_train.size, y_test.size))

    pipeline = Pipeline([
        ('scl', StandardScaler()),
        # ('lda', LinearDiscriminantAnalysis()),
        ('clf', SVC(probability=True))
    ])

    # GridSearch
    param_grid = [{'clf__kernel': ['linear', 'rbf'],
                   'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'clf__gamma': np.logspace(-2, 2, 5),
                   # 'lda__n_components': range(2, 17)
                   }]

    estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', verbose=True)

    logging.info('Training model...')
    start = timeit.default_timer()

    model = estimator.fit(X_train, y_train)
    
    stop = timeit.default_timer()
    logging.info('Time taken: {0}'.format(stop - start))

    y_pred = model.predict(X_test)

    perf = {'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'f1': f1_score(y_test, y_pred, average='macro'),
            # 'summary': classification_report(y_test, y_pred)
            }

    logging.info(perf)

    return perf, model.best_params_, model.best_estimator_
print y

pref, params, estimator = train(X, y)
save_path = './'

# Save performances
with open(os.path.join(save_path, 'performance.json'), 'w') as fp:
    json.dump(perf, fp)

# Save parameters
with open(os.path.join(save_path, 'parameters.json'), 'w') as fp:
    json.dump(params, fp)

# Save model
with open(os.path.join(save_path, 'model.pkl'), 'wb') as fp:
    pickle.dump(estimator, fp)


