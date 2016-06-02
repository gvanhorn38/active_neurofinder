'''
First pass at some active learners
'''

import random

class ActiveLearner():
    '''
    Base class for all active learners.
    '''

    def __init__(base_learner, unlabeled_data=None):
        '''
        base_learner should be a subclass of BaseLearner

        unlabeled_data should be an array of tuples (int id, video)  #TODO: Maybe it would be smarter to pass around .tiff filepaths and let BaseLearner do the accessing?
        '''
        self.base_learner = base_learner
        selt.unlabeled_data = unlabeled_data

    def predict(examples):
        '''
        Return neurofinder output for examples. Basically a wrapper around base_learner's predict().

        Args:
            examples: np.array of videos to have their neuron found

        Returns:
            np.array of strings in neurofinder json format
        '''
        #TODO: rewrite this to work with 
        return self.base_learner.predict(examples)

    def make_query():
        '''
        Prompts Active Learner to ask for a label

        Returns:
            example in self.unlabeled_data for which the learner wants a label  
        '''
        pass
        
    
class RandomLearner(ActiveLearner):
    '''
    Requests labels at random (for benchmarking) 
    '''
    
    def make_query():
        #TODO: rewrite this to work with whatever the data format is
        return random.choice(unlabeled_data)[0]


class LeastConfidenceLearner(ActiveLearner):
    '''
    Requests labels via Least Confidence Uncertainty Sampling (Lewis and Catlett, 1994)
    '''

    def make_query():
        '''
        Return example with class prediction closest to 0.5 (i.e. returns example x minimizing abs(0.5 - softmax_prediction(x)))
        '''

class InformationDenseLearner(ActiveLearner):
    
    def make_query():
        '''
        Ask for minimizer of (uncertainty(x))*(similarity(x))^b
        '''


