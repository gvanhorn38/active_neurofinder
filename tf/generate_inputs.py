import numpy as np
import random
import tensorflow as tf

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
  

def write_examples(features, labels, output_path):

  writer = tf.python_io.TFRecordWriter(output_path)

  for feature, label in zip(features, labels):
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'feature': _float_feature(feature),
        'label': _int64_feature([label]),
      }
    ))

    writer.write(example.SerializeToString())

  writer.close()

def generate_dataset(num_samples, num_positives, input_size, num_steps, output_path):
  """
  num_samples: number of "patches"
  num_positives: number of positive examples
  input_size: size of one feature vector from one "frame"
  num_steps: number of "frames" to concatenate
  """
  
  dataset = np.zeros([num_samples, input_size * num_steps], dtype=float)
  
  positive_class_indices = random.sample(xrange(num_samples), num_positives)
  labels = np.zeros(num_samples, dtype=int)
  labels[positive_class_indices] = 1
  
  
  # Make one index of the positive classes 1
  feature_indices = range(input_size * num_steps)
  for index in positive_class_indices:
    flag_index = random.choice(feature_indices)
    dataset[index, flag_index] = 1
  
  # sanity check
  print "Number of non-zeros in dataset: %d" % (np.nonzero(dataset)[0].shape[0],)
  print "Number of non-zeros in labels: %d" % (np.nonzero(labels)[0].shape[0],)
  print "Dataset and labels make sense: %s" % (np.all(np.nonzero(dataset)[0] == np.nonzero(labels)[0]),)
  
  write_examples(dataset, labels, output_path)