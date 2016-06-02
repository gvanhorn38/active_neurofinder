'''
BaseLearner template for people to subclass if they want
'''

import tensorflow as tf


class RNNModel():
  
  def __init__(self, is_training, cfg):
    
    self.batch_size
  
  def assign_learning_rate(self, session, lr_value):
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
  def cost(self):
    return self._cost
  
  @property
  def final_state(self):
    return self._final_state
  
  @property
  def lr(self):
    return self._lr
  
  @property
  def train_op(self):
    return self._train_op


class RNNLearner:
  '''
  Recurrent Neural Network Learner.
  '''

  def __init__(self, **params):
    # To store whatever hyperparameters, video info, etc. that's necessary
    self.params = params
    
    self.graph = tf.get_default_graph()
    self.session = tf.get_default_session()
    
    with self.graph.as_default(), self.session.as_default():
      
      
      with tf.variable_scope("model", reuse=None):
        self.batch_train_model = 
      
      with tf.variable_scope("model", resuse=True):
        self.update_train_model = 
      
      with tf.variable_scope("model", reuse=True):
        self.test_model = 
      
      tf.initialize_all_variables.run()

  def batch_train(self, examples, ground_truth):
    '''
    Train the model on a set of examples.

    Args:
        examples: np.array of videos in...#TODO: pick format
        ground_truth: np.array of neurofinder-format json strings, one for each video
    '''
    
    with self.graph.as_default(), self.session.as_default():
        
        
        

  def update(self, example, ground_truth):
    '''
    Update the model based on one new example.

    Args:
        example: single video example
        ground_truth: neurofinder-format json string 
    '''
    
    with self.graph.as_default(), self.session.as_default():
      
      
      
  #TODO: Decide whether to combine batch_train() and update() into one method

  def predict(self, examples):
    '''
    Make predictions about locations of neurons in a series of training examples
    
    Args:
        examples: np.array of videos

    Returns:
        #TODO: figure this out - I'm not sure what people's models do at this point
    '''
    
    with self.graph.as_default(), self.session.as_default():