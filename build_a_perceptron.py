# ----------
# 
# In this exercise, you will put the finishing touches on a perceptron class.
#
# Finish writing the activate() method by using np.dot to compute signal
# strength and then add in a threshold for perceptron activation.
#
# ----------


import numpy as np


class Perceptron:
    """
    This class models an artificial neuron with step activation function.
    """

    def __init__(self, weights = np.array([1]), threshold = 0):
        """
        Initialize weights and threshold based on input arguments. Note that no
        type-checking is being performed here for simplicity.
        """
        self.weights = weights
        self.threshold = threshold
    
    def activate(self,inputs):
        """
        Takes in @param inputs, a list of numbers equal to length of weights.
        @return the output of a threshold perceptron with given inputs based on
        perceptron weights and threshold.
        """ 

        # INSERT YOUR CODE HERE
        totalSum = np.dot(self.weights,inputs)
        
        print(totalSum)
        
        if totalSum > self.threshold:
            result=1
        else:
            result=0

        # TODO: calculate the strength with which the perceptron fires

        # TODO: return 0 or 1 based on the threshold
            
        return result


def test():
    """
    A few tests to make sure that the perceptron class performs as expected.
    Nothing should show up in the output if all the assertions pass.
    """
    p1 = Perceptron(np.array([1, 2]), 0.)
    assert p1.activate(np.array([ 1,-1])) == 0 # < threshold --> 0
    assert p1.activate(np.array([-1, 1])) == 1 # > threshold --> 1
    assert p1.activate(np.array([ 2,-1])) == 0 # on threshold --> 0

if __name__ == "__main__":
    test()