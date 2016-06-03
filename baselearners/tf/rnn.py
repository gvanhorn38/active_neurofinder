
import numpy as np
import os
import tensorflow as tf
from tensorflow.models.rnn import rnn
import time

def construct_windows(feature, label, input_length, num_steps, stride):
  
  feature_length = feature.shape[0]
  num_frames = feature_length / input_length
  window_length = input_length * num_steps
  
  windows = []
  labels = []
  for i in range(0, num_frames, stride):
    start_index = i * input_length
    end_index = (i + num_steps) * input_length
    if end_index > feature_length :
      break
    x = feature[start_index:end_index]
    windows.append(x)
    labels.append(label)

  windows = np.array(windows)
  labels = np.array(labels)
  
  return [windows, labels]

def add_inputs(tfrecords, feature_size,
  build_windows=False, input_length=None, num_steps=None, stride=None,
  num_epochs=None, 
  batch_size=10,
  num_threads=2,
  capacity=100,
  min_after_dequeue=50):
  """
  We assume that the features are a raveled version of a patch for the entire
  video sequence. At train time, we will want to limit the number of back prop steps. 
  So we will need to create "windows" of the videos, each of size num_steps (i.e. the 
  number of frames). The stride size dictates how much overlap the individual windows
  will have with each other.  
  
  At test time, we don't need to do back prop, so we can just loop through all frames of
  the video. 
  """
  
  with tf.name_scope('inputs'):

    # Have a queue that produces the paths to the .tfrecords
    filename_queue = tf.train.string_input_producer(
      tfrecords,
      num_epochs=num_epochs
    )

    # Construct a Reader to read examples from the .tfrecords file
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
    #print "Feature shape"
    #print feature.get_shape().as_list()
    
    label = tf.cast(example['label'], tf.int32)
    #print "Label shape"
    #print label.get_shape().as_list()
    
    if build_windows:
      
      window_data = tf.py_func(construct_windows, [feature, label, input_length, num_steps, stride], [tf.float32, tf.int32], name="construct_windows")
      
      window_features = window_data[0]
      window_labels = window_data[1]
      
      # Trick to get tensorflow to set the shapes so that the enqueue process works
      window_features.set_shape(tf.TensorShape([tf.Dimension(None), input_length * num_steps]))
      window_features = tf.expand_dims(window_features, 0)
      window_features = tf.squeeze(window_features, [0])
      num_windows = window_features.get_shape().as_list()[0]
      window_labels.set_shape(tf.TensorShape([tf.Dimension(num_windows)]))
      
      features, sparse_labels = tf.train.shuffle_batch(
          [window_features, window_labels],
          batch_size=batch_size,
          num_threads=num_threads,
          capacity= capacity, #batch_size * (num_threads + 2),
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue= min_after_dequeue, # 3 * batch_size,
          enqueue_many=True
        )
      
    else:
    
      features, sparse_labels = tf.train.shuffle_batch(
          [feature, label],
          batch_size=batch_size,
          num_threads=num_threads,
          capacity= capacity, #batch_size * (num_threads + 2),
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue= min_after_dequeue, # 3 * batch_size,
        )
    
  return features, sparse_labels


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  
  with tf.device('/cpu:0'):
    #var = tf.get_variable(name, shape=shape, initializer=initializer)
    var = tf.Variable(initial_value = initializer(shape), name=name)
  return var

def _variable_with_weight_decay(name, shape, initializer, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """

  var = _variable_on_cpu(name, shape, initializer)
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def add_loss(graph, logits, labels_sparse, scale=None):

  with graph.name_scope('loss'):
    batch_size, num_classes = logits.get_shape().as_list()

    labels_dense = tf.sparse_to_dense(
      sparse_indices = tf.transpose(
        tf.pack([tf.range(batch_size), labels_sparse])
      ),
      output_shape = [batch_size, num_classes],
      sparse_values = np.ones(batch_size, dtype='float32')
    )

    loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels_dense)

    loss = tf.reduce_mean(loss, name='loss')

    if scale != None:
      loss = tf.mul(loss, scale)

    tf.add_to_collection('losses', loss)

  return loss

def build(graph, input, num_steps, hidden_size, num_layers, num_classes, is_training):
  """
  
  num_steps: the number of unrolled steps of LSTM
  hidden_size: the number of LSTM units
  
  """
  
  input_shape = input.get_shape().as_list()
  batch_size = input_shape[0]
  
  # Add the GRU Cell
  with graph.name_scope("rnn") as scope:
  
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size, input_size=hidden_size)
  
    if is_training:
      gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=1.0)
  
    cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    # A length T list of inputs, each a tensor of shape [batch_size, input_size].
    inputs = [tf.squeeze(input_) for input_ in tf.split(1, num_steps, input)]
    
    print "Inputs to RNN Cell shape:"
    print "[%d, %s]" % (len(inputs), inputs[0].get_shape().as_list())
    
    outputs, state = rnn.rnn(cell, inputs, initial_state = initial_state)
    
    # [num_steps * batch_size, hidden_size]
    #features = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
    features = state
    
    print "Outputs from RNN Cell shape:"
    print features.get_shape().as_list()
    
  # Add Softmax
  with graph.name_scope("softmax") as scope:

    weights = _variable_with_weight_decay(
      name='weights',
      shape=[hidden_size, num_classes],
      initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
      wd=4e-5
    )
    graph.add_to_collection('softmax_params', weights)
     
    biases = _variable_on_cpu(
      name='biases', 
      shape=[num_classes], 
      initializer=tf.constant_initializer(0.0)
    )
    graph.add_to_collection('softmax_params', biases)

    softmax_linear = tf.nn.xw_plus_b(features, weights, biases, name="logits") 
  
  
  return softmax_linear

def train(tfrecords, save_dir, cfg):
  
  graph = tf.get_default_graph()

  sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement = True,
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.9
    )
  )
  
  max_iterations = cfg.max_iterations
  batch_size= cfg.batch_size
  input_size = cfg.input_size
  num_steps = cfg.num_steps
  frame_stride = cfg.frame_stride
  feature_size = cfg.feature_size
  num_layers = cfg.num_layers 
  
  # [batch_size, input_size * num_steps]
  #input = tf.placeholder(tf.float32, [batch_size, input_size * num_steps])
  # [batch_size * num_steps]
  #labels_sparse = tf.placeholder(tf.int32, [batch_size * num_steps])
  
  inputs, labels_sparse = add_inputs(tfrecords, feature_size=feature_size,
    build_windows=True, input_length=input_size, num_steps=num_steps, stride=frame_stride,
    num_epochs=None, 
    batch_size=batch_size,
    num_threads=2,
    capacity=1000,
    min_after_dequeue=100
  )
  
  print "Inputs shape"
  print inputs.get_shape().as_list()
  
  print "Labels Sparse shape"
  print labels_sparse.get_shape().as_list()
  
  logits = build(graph, inputs, 
    num_steps = num_steps,
    hidden_size = input_size,
    num_layers = num_layers,
    num_classes = 2,
    is_training = True
  )
  loss = add_loss(graph, logits, labels_sparse)
  
  global_step = tf.Variable(0, name='global_step', trainable=False)
  
  learning_rate = tf.train.exponential_decay(
    learning_rate=1.,
    global_step=global_step,
    decay_steps=1000,
    decay_rate=0.5,
    staircase=True
  )

  # Create the optimizer
  optimizer = tf.train.RMSPropOptimizer(
    learning_rate = learning_rate,
    decay = 0.9, # Parameter setting from the arxiv paper
    epsilon = 1.0 #Parameter setting from the arxiv paper
  )

  # Compute the gradients using the loss
  gradients = optimizer.compute_gradients(loss)
  # Apply the gradients
  optimize_op = optimizer.apply_gradients(
    grads_and_vars = gradients,
    global_step = global_step
  )
  
  fetches = [loss, learning_rate, optimize_op]
  
  print_str = ', '.join([
    'Step: %d',
    'Loss: %.4f',
    'Learning Rate: %.5f', 
    'Time/image (ms): %.1f'
  ])
  
  coord = tf.train.Coordinator()
  
  saver = tf.train.Saver()
  
  with tf.Session(graph=graph, config=sess_config) as sess:
    # make sure to initialize all of the variables
    tf.initialize_all_variables().run()

    # launch the queue runner threads
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
      step = global_step.eval()
      while step < max_iterations:
        if coord.should_stop():
          break

        t = time.time()
        fetched = sess.run(fetches)
        dt = time.time() - t

        # increment the global step counter
        step = global_step.eval()

        print print_str % (step, fetched[0], fetched[1], (dt / batch_size) * 1000)
    
    except Exception as e:
      # Report exceptions to the coordinator.
      coord.request_stop(e)
     
    saver.save(
      sess=sess,
      save_path= os.path.join(save_dir, 'model'),
      global_step=step
    )

  # When done, ask the threads to stop. It is innocuous to request stop twice.
  coord.request_stop()
  # And wait for them to actually do it.
  coord.join(threads) 
  
  
  
def test(tfrecords, model_path, cfg):
   
  graph = tf.get_default_graph()

  sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement = True,
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.9
    )
  )
  
  max_iterations = cfg.max_iterations
  batch_size= cfg.batch_size
  input_size = cfg.input_size
  num_steps = cfg.num_steps
  frame_stride = cfg.frame_stride
  feature_size = cfg.feature_size 
  
  # [batch_size, input_size * num_steps]
  #input = tf.placeholder(tf.float32, [batch_size, input_size * num_steps])
  # [batch_size * num_steps]
  #labels_sparse = tf.placeholder(tf.int32, [batch_size * num_steps])
  
  inputs, labels_sparse = add_inputs(tfrecords, feature_size,
    num_epochs=1, 
    batch_size=batch_size,
    num_threads=2,
    capacity=100,
    min_after_dequeue=0
  )
  
  #print "Inputs shape"
  #print inputs.get_shape().as_list()
  
  #print "Labels Sparse shape"
  #print labels_sparse.get_shape().as_list()
  
  logits = build(graph, inputs, 
    num_steps = num_steps,
    hidden_size = input_size,
    num_layers = 1,
    num_classes = 2,
    is_training = False
  )
  
  top_k_op = tf.nn.in_top_k(logits, labels_sparse, 1)
  
  fetches = [top_k_op]
  
  print_str = ', '.join([
    'Evaluated batch %d',
    'Total Number Correct: %d / %d',
    'Current Precision: %.3f',
    'Time/image (ms): %.1f'
  ])
  
  coord = tf.train.Coordinator()
  
  saver = tf.train.Saver(tf.trainable_variables())
  
  with tf.Session(graph=graph, config=sess_config) as sess:
    # make sure to initialize all of the variables
    tf.initialize_all_variables().run()

    # launch the queue runner threads
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
      
      saver.restore(sess, model_path)
      
      true_count = 0.0  # Counts the number of correct predictions.
      total_sample_count = 0
      step = 0
      while not coord.should_stop():

        t = time.time()
        outputs = sess.run(fetches)
        dt = time.time()-t

        predictions = outputs[0]
        true_count += np.sum(predictions)

        print print_str % (
          step,
          true_count,
          (step + 1) * batch_size,
          true_count / ((step + 1.) * batch_size),
          dt/batch_size*1000
        )

        step += 1
        total_sample_count += batch_size
    
    except Exception as e:
      # Report exceptions to the coordinator.
      coord.request_stop(e)

  # When done, ask the threads to stop. It is innocuous to request stop twice.
  coord.request_stop()
  # And wait for them to actually do it.
  coord.join(threads) 
  
    
  