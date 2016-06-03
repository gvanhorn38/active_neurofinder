'''
BaseLearner template for people to subclass if they want
'''
import copy
from easydict import EasyDict
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn
import time

def default_config():
  return EasyDict({
    
    'batch_size' : 30,
    
    # RNN Model Params
    'input_size' : 225, # Size of a feature from one video frame
    'hidden_size' : 225, # Number of hidden units
    'keep_prob' : 1, # Dropout keep probability
    'num_layers' : 1, # Number of layers for the rnn
    
    # How many unrolls to do for the back prop 
    'num_steps' : 250, 
    
    # Learning Rate stuff
    'lr' : 1.,
    'lr_decay_steps' : 1000,
    'lr_decay_rate' : 0.5,
    'lr_staircase' : True,
    
    # Max training iterations
    'max_iterations' : 5000,
    
    # The stride length when splitting a video into multiple smaller segments
    'frame_stride' : 1000,
    
    # Queue parameters when doing the batch training
    'num_threads' : 4,
    'capacity' : 1000,
    'min_after_dequeue' : 100
  
  })

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
    var = tf.get_variable(name, shape=shape, initializer=initializer)
    #var = tf.Variable(initial_value = initializer(shape), name=name)
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


class RNNModel():
  
  def __init__(self, is_training, cfg):
    
    self.batch_size = batch_size = cfg.batch_size
    self.num_steps = num_steps = cfg.num_steps
    
    hidden_size = cfg.hidden_size
    input_size = cfg.input_size
    num_classes = 2
    
    # These will get filled in for us
    self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps * input_size])
    self._targets = tf.placeholder(tf.int32, [batch_size])
    
    graph = tf.get_default_graph()
    
    # Add the GRU cell
    with graph.name_scope("rnn") as scope:
  
      gru_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size, input_size=input_size)
  
      if is_training:# and cfg.keep_prob < 1:
        gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=cfg.keep_prob)
  
      cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * cfg.num_layers)
      
      self._initial_state = cell.zero_state(batch_size, tf.float32)
      
      #print self._initial_state.get_shape().as_list()
      
      
      # A length T list of inputs, each a tensor of shape [batch_size, input_size].
      inputs = [tf.squeeze(input_) for input_ in tf.split(1, num_steps, self._input_data)]
      #print len(inputs)
      if batch_size == 1:
        inputs = [tf.expand_dims(t, 0) for t in inputs]

      #print inputs[0].get_shape().as_list()
      
      outputs, state = rnn.rnn(cell, inputs, initial_state = self._initial_state)
    
      # [num_steps * batch_size, hidden_size]
      #features = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
      features = outputs[-1]
      self._final_state = state
      
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

      self._logits = tf.nn.xw_plus_b(features, weights, biases, name="logits") 
    
    # Add loss
    with graph.name_scope('loss'):

      labels_dense = tf.sparse_to_dense(
        sparse_indices = tf.transpose(
          tf.pack([tf.range(batch_size), self._targets])
        ),
        output_shape = [batch_size, num_classes],
        sparse_values = np.ones(batch_size, dtype='float32')
      )

      loss = tf.nn.softmax_cross_entropy_with_logits(self._logits, labels_dense)

      self._loss = loss = tf.reduce_mean(loss, name='loss')

      tf.add_to_collection('losses', loss)
    
    if not is_training:
      return
    
    self._lr = tf.Variable(0.0, trainable=False)
    
    # Create the optimizer
    optimizer = tf.train.RMSPropOptimizer(
      learning_rate = self._lr,
      decay = 0.9, 
      epsilon = 1.0 
    )

    # Compute the gradients using the loss
    gradients = optimizer.compute_gradients(loss)
    
    # Apply the gradients
    self._train_op = optimizer.apply_gradients(
      grads_and_vars = gradients
      #global_step = global_step
    )
    
  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))
  
  @property
  def input_data(self):
    return self._input_data
  
  @property
  def targets(self):
    return self._targets
  
  @property
  def initial_state(self):
    return self._initial_state
  
  @property
  def loss(self):
    return self._loss
  
  @property
  def final_state(self):
    return self._final_state
  
  @property
  def lr(self):
    return self._lr
  
  @property
  def train_op(self):
    return self._train_op
  
  @property
  def logits(self):
    return self._logits


class RNNLearner:
  '''
  Recurrent Neural Network Base Learner.
  '''

  def __init__(self, cfg):
    # To store whatever hyperparameters, video info, etc. that's necessary
    self.params = cfg
    
    self.graph = tf.Graph()#tf.get_default_graph()
    self.session = tf.Session(graph=self.graph)
    
    with self.graph.as_default(), self.session.as_default():
      
      with tf.variable_scope("model", reuse=None):
        self.batch_train_model = RNNModel(is_training=True, cfg=cfg)
      
      self.restore_vars = tf.trainable_variables()
      
      cfg_update = copy.copy(cfg)
      cfg_update.batch_size = 1
      with tf.variable_scope("model", reuse=True):
        self.update_train_model = RNNModel(is_training=True, cfg=cfg_update)
      
      cfg_test = copy.copy(cfg)
      cfg_test.batch_size = 1
      cfg_test.num_steps = 3048 # GVH: Hack to get an experiment done
      with tf.variable_scope("model", reuse=True):
        self.test_model = RNNModel(is_training=False, cfg=cfg_test)
      
      tf.initialize_all_variables().run()
  
  def restore(self, model_path):
    with self.graph.as_default(), self.session.as_default():
      
      # quick patch to make this match the rnn.py model
      var_dict = {}
      for v in self.restore_vars:
        n = v.name[6:]
        if n == 'weights:0' or n == 'biases:0':
          n = 'softmax/' + n
        var_dict[n[:-2]] = v
      
      saver = tf.train.Saver(var_dict)
      saver.restore(self.session, model_path)
  
  def save(self, model_path):
    with self.graph.as_default(), self.session.as_default():
      saver = tf.train.Saver(self.restore_vars)
      saver.save(
        sess=self.session,
        save_path=model_path
      )
  
  def batch_train(self, examples, ground_truth):
    '''
    Train the model on a set of examples.

    Args:
        examples: np.array of videos in...#TODO: pick format
        ground_truth: np.array of neurofinder-format json strings, one for each video
        
        Additional args: 
          max_iterations
          learning rate
          
          batch_size
          input_length 
          num_steps
          stride
          
          num_threads
          
    '''
    
    cfg = self.params
    
    with self.graph.as_default(), self.session.as_default():
      
      global_step = tf.Variable(0, name='global_step', trainable=False)
      step_incr = tf.count_up_to(global_step, cfg.max_iterations)
      
      lr = tf.train.exponential_decay(
        learning_rate=cfg.lr,
        global_step=global_step,
        decay_steps=cfg.lr_decay_steps,
        decay_rate=cfg.lr_decay_rate,
        staircase=cfg.lr_staircase
      )  
      
      
      feature, label = tf.train.slice_input_producer(
        [examples, ground_truth],
        num_epochs= None,
        shuffle=True,
        seed = None,
        capacity = 1000
      )
      
      feature = tf.to_float(feature)
      
      window_data = tf.py_func(construct_windows, [feature, label, cfg.input_size, cfg.num_steps, cfg.frame_stride], [tf.float32, tf.int32], name="construct_windows")
      
      window_features = window_data[0]
      window_labels = window_data[1]
      
      # Trick to get tensorflow to set the shapes so that the enqueue process works
      window_features.set_shape(tf.TensorShape([tf.Dimension(None), cfg.input_size * cfg.num_steps]))
      window_features = tf.expand_dims(window_features, 0)
      window_features = tf.squeeze(window_features, [0])
      num_windows = window_features.get_shape().as_list()[0]
      window_labels.set_shape(tf.TensorShape([tf.Dimension(num_windows)]))
      
      features, sparse_labels = tf.train.shuffle_batch(
        [window_features, window_labels],
        batch_size=cfg.batch_size,
        num_threads=cfg.num_threads,
        capacity= cfg.capacity, #batch_size * (num_threads + 2),
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue= cfg.min_after_dequeue, # 3 * batch_size,
        enqueue_many=True
      )
      
      
      
      print_str = ', '.join([
        'Step: %d',
        'Loss: %.4f',
        'Learning Rate: %.5f', 
        'Time/image (ms): %.1f'
      ])
      
      coord = tf.train.Coordinator()
      
      tf.initialize_variables([global_step]).run()
      
      try:
        
        threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
        
        step = global_step.eval()
        while step < cfg.max_iterations:
          if coord.should_stop():
            break
          self.batch_train_model.assign_lr(self.session, lr.eval())
          t = time.time()
          fetched = self.session.run(
            [self.batch_train_model.loss, lr, self.batch_train_model.train_op, step_incr],
            {
              self.batch_train_model.input_data : features.eval(),
              self.batch_train_model.targets : sparse_labels.eval()
            }
          )
          dt = time.time() - t
          
          step = global_step.eval()

          print print_str % (step, fetched[0], fetched[1], (dt / cfg.batch_size) * 1000)

      except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop(e)

      # When done, ask the threads to stop. It is innocuous to request stop twice.
      coord.request_stop()
      # And wait for them to actually do it.
      coord.join(threads)     
        
        

  def update(self, example, ground_truth):
    '''
    Update the model based on one new example.

    Args:
        example: single video example
        ground_truth: neurofinder-format json string 
        
    Additional Args:
      lr
          
    '''
    cfg = self.params
    lr = 1. # ????
    
    print_str = ', '.join([
      'Step: %d',
      'Loss: %.4f',
      'Learning Rate: %.5f', 
      'Time/image (ms): %.1f'
    ])
    
    with self.graph.as_default(), self.session.as_default():
      
      features, labels = construct_windows(example, ground_truth, cfg.input_size, cfg.num_steps, cfg.frame_stride)
      
      self.update_train_model.assign_lr(self.session, lr)
      
      for i in range(features.shape[0]):
        
        feature = features[i]
        label = labels[i]
        
        t = time.time()      
        fetched = self.session.run(
          [self.update_train_model.loss, self.update_train_model.train_op],
          {
            self.update_train_model.input_data : [feature],
            self.update_train_model.targets : [label]
          }
        )
        dt = time.time() - t
        
        print print_str % (i, fetched[0], lr, (dt / 1.) * 1000)
        
  
  # we'll need batches here. 
  def batch_predict(self, examples):
    '''
    Make predictions about locations of neurons in a series of training examples
    
    Args:
        examples: np.array of videos

    Returns:
        #TODO: figure this out - I'm not sure what people's models do at this point
    '''
    
    cfg = self.params
    all_logits = []
    
    print_str = ', '.join([
      'Step: %d',
      'Time/image (ms): %.1f'
    ])
    
    with self.graph.as_default(), self.session.as_default():
      
      initial_state = self.test_model.initial_state.eval()
      
      for i in range(examples.shape[0]):
        
        example = examples[i]
      
        # we want to loop over each frame 
        #features = example.reshape((-1, cfg.input_size))
        state = initial_state
        t = time.time()   
        #for feature in features:
        logits, state = self.session.run(
          [self.test_model.logits, self.test_model.final_state],
          {
            self.test_model.input_data : [example],#[feature],
            self.test_model.initial_state : state
          }
        )
        dt = time.time() - t
        
        print print_str % (i, (dt / 1.) * 1000)
        
        all_logits.append(logits)
        
    return all_logits
            
  def predict(self, example):
    
    cfg = self.params
    
    with self.graph.as_default(), self.session.as_default():
      
      # we want to loop over each frame 
      features = example.reshape((-1, cfg.input_size))
      state = self.test_model.initial_state.eval()
      for feature in features:
        logits, state = self.session.run(
          [self.test_model.logits, self.test_model.final_state],
          {
            self.test_model.input_data : [feature],
            self.test_model.initial_state : state
          }
        )
      
      return logits