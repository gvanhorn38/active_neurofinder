import load_dataset
import rnn_learner
from collections import Counter
from sklearn import metrics
import cPickle as pickle

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
  'Active Precision: %.3f',
  'Test Precision: %.3f'
])

results = []
query_counts = Counter()
test_increment = 25

for i in range(1, 2*active_labels.shape[0]):
  predictions = base_learner.batch_predict(active_features)
  query = np.argmin(np.abs(predictions[:,0] - predictions[:,1]))
  query_counts[query] += 1

  base_learner.update(active_features[query], active_labels[query])
  
  if i % test_increment == 0: 
    
    active_prediction = np.argmax(base_learner.batch_predict(active_features), axis=1)
    active_accuracy = metrics.accuracy_score(active_labels, active_prediction)
    active_precision, active_recall, active_f1, support = (tuple(x) for x in metrics.precision_recall_fscore_support(active_labels, active_prediction))
    
    test_prediction = np.argmax(base_learner.batch_predict(test_features), axis=1)
    test_accuracy = metrics.accuracy_score(test_labels, test_prediction)
    test_precision, test_recall, test_f1, support = (tuple(x) for x in metrics.precision_recall_fscore_support(test_labels, test_prediction))
  
    results.append((i, query_counts, active_accuracy, active_f1, active_precision, active_recall, test_accuracy, test_f1, test_precision, test_recall))
    
    print print_str % (i, active_accuracy, test_accuracy)
    
    
output_path = '/home/gvanhorn/Desktop/active_results.pkl'
with open(output_path, 'w') as f:
  pickle.dump(results, f)




#########################
  
logits = r.batch_predict(test_features)

num_correct = 0.
for gt_label, log_probs in zip(test_labels, logits):
  pred_label = np.argmax(log_probs)
  
  if gt_label == pred_label:
    num_correct += 1

print "Accuracy: %0.3f" % (num_correct / logits.shape[0])
  
