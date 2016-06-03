'''
Active learning experiments.
'''
import numpy as np
import pickle
from sklearn import metrics
from baselearner.tf import load_dataset, rnn_learner
from collections import Counter

def uncertainty_sampling_active_learner(base_learner, active_dataset_path, test_dataset_path, test_increment, output_path):
    '''
    Tests how much labeling effort is saved by the use of the active learner on the input dataset

    Args:
        active_learner: ActiveLearner instance 
        output_path: place to save pickles results (should probably end in .pkl)

    Returns:
        list of tuples (# of queries made, number of times each example was queried, accuracy, f1, precision, recall)
    '''
    active_features, active_labels = load_dataset.load_dataset([active_dataset_path])
    test_features, test_labels = load_dataset.load_dataset([test_dataset_path])

    results = []
    query_counts = Counter()

    for i in range(1, 2*len(train_labels)):
        predictions = base_learner.batch_predict(active_features)
        query = np.argmin(np.abs(output[:,0] - output[:,1]))
        query_counts[query] += 1

        base_learner.update(active_features[query], active_labels[query])

        if i % test_increment == 0: 
            active_prediction = np.argmax(base_learner.batch_predict(active_features), axis=1)
            accuracy = metrics.accuracy_score(active_labels, active_prediction)
            active_precision, active_recall, active_f1, support = (tuple(x) for x in metrics.precision_recall_fscore_support(active_labels, active_prediction))
            test_prediction = np.argmax(base_learner.batch_predict(test_features), axis=1)
            test_accuracy = metrics.accuracy_score(test_labels, test_prediction)
            test_precision, test_recall, test_f1, support = (tuple(x) for x in metrics.precision_recall_fscore_support(test_labels, test_prediction))
          
            results.append((i, query_counts, active_accuracy, active_f1, active_precision, active_recall, test_accuracy, test_precision, test_recall, test_f1))
        

    # Write to file
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

