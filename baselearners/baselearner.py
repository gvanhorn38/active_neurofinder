'''
BaseLearner template for people to subclass if they want
'''

class BaseLearner:
    '''
    Base class for all visual classifiers used by the active learner.
    '''

    def __init__(self, **params):
        # To store whatever hyperparameters, video info, etc. that's necessary
        self.params = params

    def batch_train(self, examples, ground_truth):
        '''
        Train the model on a set of examples.
    
        Args:
            examples: np.array of videos in...#TODO: pick format
            ground_truth: np.array of neurofinder-format json strings, one for each video
        '''
        pass

    def update(self, example, ground_truth):
        '''
        Update the model based on one new example.

        Args:
            example: single video example
            ground_truth: neurofinder-format json string 
       '''

    #TODO: Decide whether to combine batch_train() and update() into one method

    def predict(self, examples):
        '''
        Make predictions about locations of neurons in a series of training examples
        
        Args:
            examples: np.array of videos

        Returns:
            #TODO: figure this out - I'm not sure what people's models do at this point
        '''
