from periscope import Network
import lasagne
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, DropoutLayer
from lasagne.layers.normalization import batch_norm
from lasagne.init import HeUniform, Constant
from lasagne.nonlinearities import identity
import numpy as np
from collections import OrderedDict

# Initialize batchnorm biases with -0.25, -0.5, -0.75, -1.0
# and do not update betas
# Fixed application of batchnorm. 23669156 params.
class Ww4bn3fixed(Network):

    def init_constants(self):
        super(Ww4bn3fixed, self).init_constants()
        self.crop_size = (96, 96)
        self.batch_size = 64
        self.learning_rates = np.concatenate((
            [0.005] * 10,
            [0.0005] * 10,
            [0.00005] * 10,
            [0.000005] * 10
        ))

    def hidden_layers(self, network, **kwargs):
        # 1st. Data size 96->96
        network = Conv2DLayer(network, 64, (3, 3), pad='same',
            W=HeUniform('relu'))
        network = batch_norm(network, beta=Constant(-0.25), gamma=None)
        # 2nd. Data size 96->96
        network = Conv2DLayer(network, 64, (3, 3), pad='same',
            W=HeUniform('relu'))
        network = batch_norm(network, beta=Constant(-0.25), gamma=None)

        # Max pool. Data size 96->48
        network = MaxPool2DLayer(network, (2, 2), stride=2)

        # 3rd. Data size 48->48
        network = Conv2DLayer(network, 128, (3, 3), pad='same',
            W=HeUniform('relu'))
        network = batch_norm(network, beta=Constant(-0.5), gamma=None)
        # 4th. Data size 48->48
        network = Conv2DLayer(network, 128, (3, 3), pad='same',
            W=HeUniform('relu'))
        network = batch_norm(network, beta=Constant(-0.5), gamma=None)

        # Max pool. Data size 48->24
        network = MaxPool2DLayer(network, (2, 2), stride=2)

        # 5th. Data size 24->24
        network = Conv2DLayer(network, 256, (3, 3), pad='same',
            W=HeUniform('relu'), name='conv5')
        network = batch_norm(network, beta=Constant(-0.75), gamma=None)
        # 6th. Data size 24->24
        network = Conv2DLayer(network, 256, (3, 3), pad='same',
            W=HeUniform('relu'), name='conv6')
        network = batch_norm(network, beta=Constant(-0.75), gamma=None)

        # Max pool.  Data size 24->12
        network = MaxPool2DLayer(network, (2, 2), stride=2)

        # 7th. Data size 12->12
        network = Conv2DLayer(network, 512, (3, 3), pad='same',
            W=HeUniform('relu'), name='conv7')
        network = batch_norm(network, beta=Constant(-1.0), gamma=None)

        # 8th. Data size 12->12
        network = Conv2DLayer(network, 512, (3, 3), pad='same',
            W=HeUniform('relu'), name='conv8')
        network = batch_norm(network, beta=Constant(-1.0), gamma=None)

        # Max pool.  Data size 12->6
        network = MaxPool2DLayer(network, (2, 2), stride=2)

        # 9th. Data size 6->1
        network = lasagne.layers.DenseLayer(network, 1024, W=HeUniform('relu'))
        network = batch_norm(network, gamma=None)

        network = DropoutLayer(network)

        return network

    def parameter_updates(self, training_loss):
        import lasagne
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        betas = [p for p in params if p.name == 'beta']
        weights = [p for p in params if p.name != 'beta']
        updates = OrderedDict()
        updates.update(lasagne.updates.nesterov_momentum(
                training_loss,
                weights,
                learning_rate=self.learning_rate(),
                momentum=0.9))
        # Do not update betas!
        # updates.update(lasagne.updates.nesterov_momentum(
        #         training_loss,
        #         betas,
        #         learning_rate=self.learning_rate() * 8,
        #         momentum=0.9))
        return updates

    def regularize_parameters(self):
        l2 = lasagne.regularization.regularize_network_params(
            self.network, lasagne.regularization.l2) * 5e-3
        return l2
