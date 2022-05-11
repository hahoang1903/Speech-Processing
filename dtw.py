from functools import reduce
from os import path
from operator import itemgetter
from librosa import sequence
import numpy as np
from mfcc import extract_mfccs, get_mfccs
from data import generate_dtw_template

# template_path = generate_dtw_template(['19021261', '19021289', '19021367'])

# extract mfcc of templates
# extract_mfccs(template_path, 'mfcc/templates')

# extract mfcc of tests
# extract_mfccs('data/segmented/19021261', 'mfcc/19021261')

count = 0
true_count = 0

for test_label in get_mfccs('mfcc/19021261'):
    label = test_label['label']

    for test_mfcc in test_label['mfccs']:
        dtw_costs = []
        output_label = None
        count += 1

        for template_label in get_mfccs('mfcc/templates'):
            dtw_cost = np.average(
                [
                    sequence.dtw(X=test_mfcc, Y=template_mfcc,
                                 backtrack=False)[-1][-1]
                    for template_mfcc in template_label['mfccs']
                ]
            )

            dtw_costs.append(
                {'cost': dtw_cost, 'label': template_label['label']})

        min_cost = min(dtw_costs, key=itemgetter('cost'))
        print(f"true: {label}", f"real: {min_cost['label']}", sep='---')
        if (label.lower() == min_cost['label'].lower()):
            true_count += 1

print(f"{float(true_count)/float(count):.0%}")
