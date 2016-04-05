from periscope import Network
import lasagne
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, DropoutLayer
from lasagne.layers.normalization import batch_norm
from lasagne.init import HeUniform
from lasagne.nonlinearities import identity
import numpy as np

# Ww4 without batchnorm. 23665316 params.
class Ww4(Network):

    def init_constants(self):
        super(Ww4, self).init_constants()
        self.crop_size = (96, 96)
        self.batch_size = 64
        self.learning_rates = np.concatenate((
            [0.005] * 20,
            [0.0005] * 20,
            [0.00005] * 20,
            [0.000005] * 20
        ))

    def hidden_layers(self, network, **kwargs):
        # 1st. Data size 96->96
        network = Conv2DLayer(network, 64, (3, 3), pad='same',
            W=HeUniform('relu'))
        # 2nd. Data size 96->96
        network = Conv2DLayer(network, 64, (3, 3), pad='same',
            W=HeUniform('relu'))

        # Max pool. Data size 96->48
        network = MaxPool2DLayer(network, (2, 2), stride=2)

        # 3rd. Data size 48->48
        network = Conv2DLayer(network, 128, (3, 3), pad='same',
            W=HeUniform('relu'))
        # 4th. Data size 48->48
        network = Conv2DLayer(network, 128, (3, 3), pad='same',
            W=HeUniform('relu'))

        # Max pool. Data size 48->24
        network = MaxPool2DLayer(network, (2, 2), stride=2)

        # 5th. Data size 24->24
        network = Conv2DLayer(network, 256, (3, 3), pad='same',
            W=HeUniform('relu'), name='conv5')
        # 6th. Data size 24->24
        network = Conv2DLayer(network, 256, (3, 3), pad='same',
            W=HeUniform('relu'), name='conv6')

        # Max pool.  Data size 24->12
        network = MaxPool2DLayer(network, (2, 2), stride=2)

        # 7th. Data size 12->12
        network = Conv2DLayer(network, 512, (3, 3), pad='same',
            W=HeUniform('relu'), name='conv7')

        # 8th. Data size 12->12
        network = Conv2DLayer(network, 512, (3, 3), pad='same',
            W=HeUniform('relu'), name='conv8')

        # Max pool.  Data size 12->6
        network = MaxPool2DLayer(network, (2, 2), stride=2)

        # 9th. Data size 6->1
        network = lasagne.layers.DenseLayer(network, 1024, W=HeUniform('relu'))
        network = batch_norm(network, gamma=None)

        network = DropoutLayer(network)

        return network

    def regularize_parameters(self):
        l2 = lasagne.regularization.regularize_network_params(
            self.network, lasagne.regularization.l2) * 5e-3
        return l2
