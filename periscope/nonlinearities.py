import theano.tensor.nnet

# rectified linear squish.
# sets the middle 50% (or within a selected threshold) to zero.
def rectsquish(x, thresh=0.67448):
    """Double-sided rectify."""
    return lambda x: (theano.tensor.nnet.relu(x - thresh) -
                      theano.tensor.nnet.relu(-x - thresh))

