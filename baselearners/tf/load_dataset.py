import numpy as np
import tensorflow as tf

def load_dataset(tfrecords, feature_size):
  
  all_features = []
  all_labels = []
  
  graph = tf.Graph()
  session = tf.Session(graph=graph)
  
  with graph.as_default(), session.as_default():
  
    filename_queue = tf.train.string_input_producer(
      tfrecords,
      num_epochs=1
    )
  
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Parse an Example to access the Features
    example = tf.parse_single_example(
      serialized_example,
      features = {
        'feature'  : tf.FixedLenFeature([feature_size], tf.float32),
        'label' : tf.FixedLenFeature([], tf.int64)
      }
    )

    feature = example['feature']
  
    label = tf.cast(example['label'], tf.int32)
  
    coord = tf.train.Coordinator() 
  
    tf.initialize_all_variables().run()
    
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    
    try:
      while True:
        if coord.should_stop():
          break 
    
        f, l = session.run([feature, label])
        all_features.append(f)
        all_labels.append(l)
    
    except Exception as e:
      # Report exceptions to the coordinator.
      coord.request_stop(e)

    # When done, ask the threads to stop. It is innocuous to request stop twice.
    coord.request_stop()
    # And wait for them to actually do it.
    #coord.join(threads)
  
  return np.array(all_features), np.array(all_labels)