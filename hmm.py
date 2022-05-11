import logging
import os
import pickle

import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from operator import itemgetter

from mfcc import extract_mfccs, get_mfccs

logging.getLogger("hmmlearn").setLevel("CRITICAL")


def get_data_set(id):
    data_set = {}
    # for id in [name for name in os.listdir('data/segmented') if os.path.isdir(os.path.join('data/segmented', name))]:
    mfcc_dir = f"mfcc/{id}"
    if not os.path.exists(mfcc_dir):
        extract_mfccs(f'data/segmented/{id}', mfcc_dir)
    for mfcc_with_label in get_mfccs(mfcc_dir):
        label, mfccs = itemgetter('label', 'mfccs')(mfcc_with_label)
        if label not in data_set:
            data_set[label] = []
        data_set[label] = data_set[label] + mfccs

    return data_set


def train_models():
    models = {}
    tests = {}
    STATES_NUM = 3
    GMM_MIX_NUM = 3

    trans_p = 1.0 / STATES_NUM - GMM_MIX_NUM
    transmat_prior = np.array([[trans_p, 0, 0],
                               [0, trans_p, 0],
                               [0, 0, trans_p]], dtype=np.float_)
    startprob_prior = np.array([0.5, 0.5, 0], dtype=np.float_)

    data_set = get_data_set('19021261')
    for label in data_set.keys():
        train_set, tests[label] = train_test_split(
            data_set[label], test_size=0.2)
        model = hmm.GMMHMM(n_components=STATES_NUM, n_mix=GMM_MIX_NUM, transmat_prior=transmat_prior,
                           startprob_prior=startprob_prior, covariance_type='diag', n_iter=300)
        lengths = np.zeros([len(train_set), ], dtype=np.int_)

        train_set_T = [feat.T for feat in train_set]

        for m in range(len(train_set_T)):
            lengths[m] = train_set_T[m].shape[0]
        train_set_T = np.vstack(train_set_T)
        model.fit(train_set_T, lengths=lengths)
        models[label] = model

    pickle.dump(models, open('hmm_model/model_hmm.pkl', 'wb'))
    return models


if not os.path.isfile('hmm_model/model_hmm.pkl'):
    models = train_models()
else:
    models = pickle.load(open('hmm_model/model_hmm.pkl', 'rb'))
print('Finished training')

score_cnt = 0
total_cnt = 0
data = get_data_set('19021261')
tests = {}
for label in data.keys():
    _, tests[label] = train_test_split(data[label], test_size=0.35)
for label in tests.keys():
    features = tests[label]
    features_T = [feat.T for feat in features]

    for feature in features_T:
        total_cnt += 1
        score_list = {}
        for model_label in models.keys():
            score_list[model_label] = models[model_label].score(feature)

        pred = max(score_list, key=score_list.get)
        if pred == label:
            score_cnt += 1

print("Final recognition rate is %.2f" %
      (100.0*score_cnt/total_cnt), "%")
