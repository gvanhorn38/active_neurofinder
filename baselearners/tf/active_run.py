import load_dataset
import rnn_learner
from collections import Counter
from sklearn import metrics
import cPickle as pickle
import numpy as np
import random

#train_features, train_labels = load_dataset.load_dataset(['/home/gvanhorn/Desktop/neuron_trial/datasets/neurofinder.00.00.tfrecords'], 680400)

active_features, active_labels = load_dataset.load_dataset(['/home/gvanhorn/Desktop/neuron_trial/datasets/neurofinder.00.01.tfrecords'], 685800)

test_features, test_labels = load_dataset.load_dataset(['/home/gvanhorn/Desktop/neuron_trial/datasets/neurofinder.00.02.tfrecords'], 693000)

test_features = test_features[:,:685800] # GVH: Hack to make them the same length as the active features

cfg = rnn_learner.default_config()
cfg.batch_size=10
cfg.input_size = 225
cfg.hidden_size = 225
cfg.num_layers = 1
cfg.num_steps = 500
cfg.frame_stride = 250
cfg.max_iterations = 500
cfg.num_threads = 4
cfg.capacity = 1000
cfg.min_after_dequeue = 100

base_learner = rnn_learner.RNNLearner(cfg)

base_learner.restore('/home/gvanhorn/Desktop/neuron_trial/pretrained/model-5000')

#r.batch_train(train_features, train_labels)

print_str = ', '.join([
  'Step: %d',
  'Active Accuracy: %.3f',
  'Test Accuracy: %.3f'
])

results = []
query_counts = Counter()
test_increment = 5

lf = open('/home/gvanhorn/Desktop/active_neuron_log2.txt', 'w')

# Do an initial test
active_prediction = np.argmax(base_learner.batch_predict(active_features), axis=1)
active_accuracy = metrics.accuracy_score(active_labels, active_prediction)
active_precision, active_recall, active_f1, support = (tuple(x) for x in metrics.precision_recall_fscore_support(active_labels, active_prediction))

test_prediction = np.argmax(base_learner.batch_predict(test_features), axis=1)
test_accuracy = metrics.accuracy_score(test_labels, test_prediction)
test_precision, test_recall, test_f1, support = (tuple(x) for x in metrics.precision_recall_fscore_support(test_labels, test_prediction))

results.append((0, query_counts, active_accuracy, active_f1, active_precision, active_recall, test_accuracy, test_f1, test_precision, test_recall))

print print_str % (0, active_accuracy, test_accuracy)    
print >> lf, print_str % (0, active_accuracy, test_accuracy)
lf.flush()

for i in range(7, 2*active_labels.shape[0]):
  
  print "###"
  print "Starting iteration %d" % i
  
  idx = random.sample(range(active_features.shape[0]), 100)
  
  predictions = base_learner.batch_predict(active_features[idx], verbose=False)
  query = np.argmin(np.abs(predictions[:,0] - predictions[:,1]))
  #query_counts[query] += 1
  print "Querying a %d example" % active_labels[idx][query]
  
  base_learner.update(active_features[idx][query], active_labels[idx][query], lr=0.01, verbose=True)
  
  if i % test_increment == 0: 
    print "---"
    print "TESTING"
    
    # subsample the features to get a speedup
    #idx = random.sample(range(active_features.shape[0]), 30)
    
    active_prediction = np.argmax(base_learner.batch_predict(active_features, verbose=False), axis=1)
    active_accuracy = metrics.accuracy_score(active_labels, active_prediction)
    active_precision, active_recall, active_f1, support = (tuple(x) for x in metrics.precision_recall_fscore_support(active_labels, active_prediction))
    
    #idx = random.sample(range(test_features.shape[0]), 30)
    
    test_prediction = np.argmax(base_learner.batch_predict(test_features, verbose=False), axis=1)
    test_accuracy = metrics.accuracy_score(test_labels, test_prediction)
    test_precision, test_recall, test_f1, support = (tuple(x) for x in metrics.precision_recall_fscore_support(test_labels, test_prediction))
  
    results.append((i, query_counts, active_accuracy, active_f1, active_precision, active_recall, test_accuracy, test_f1, test_precision, test_recall))
    
    print print_str % (i, active_accuracy, test_accuracy)
    print "---"
    
    print >> lf, print_str % (i, active_accuracy, test_accuracy)
    lf.flush()
    
  print "End iteration %d" % i

lf.close()
    
output_path = '/home/gvanhorn/Desktop/active_results2.pkl'
with open(output_path, 'w') as f:
  pickle.dump(results, f)




#########################
  
# logits = r.batch_predict(test_features)
# 
# num_correct = 0.
# for gt_label, log_probs in zip(test_labels, logits):
#   pred_label = np.argmax(log_probs)
#   
#   if gt_label == pred_label:
#     num_correct += 1
# 
# print "Accuracy: %0.3f" % (num_correct / logits.shape[0])
  
