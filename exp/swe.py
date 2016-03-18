from periscope import Network
from periscope.layers import ZeroGrayLayer, QuickNormLayer, ZeroPreluLayer
import lasagne
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, ConcatLayer
from lasagne.layers import FeatureWTALayer, NonlinearityLayer
from lasagne.init import HeNormal
from lasagne.nonlinearities import identity, rectify
import numpy as np

# WTA layer only at the end,
# plus padding at most layers,
# plus a 2x2 convolution at the end.
class Swe(Network):

    def init_constants(self):
        super(Swe, self).init_constants()
        self.crop_size = (99, 99)
        self.learning_rates = np.logspace(-2, -5.33333, 40)

    def hidden_layers(self, network, **kwargs):
        network = ZeroGrayLayer(network)

        # 1st. Data size 99 -> 97
        # 97*97*32 = 301088, rf:3x3
        network = Conv2DLayer(network, 32, (3, 3),
            W=HeNormal('relu'), name="conv1")
        network = apply_relu_bn(network)
        # 2nd. Data size 97 -> 49
        # 49*49*64 = 153644, rf:5x5
        # 32 + 32 = 64 ch
        network = swp_gadget(network, 32, stride=2, pad=1, name="goo2")

        # 3nd. Data size 49 -> 25
        # 25*25*112 = 70000, rf:9x9
        # 64 + 48 = 112 ch
        network = swp_gadget(network, 48, stride=2, pad=1, name="goo3")

        # 4th. Data size 25 -> 13
        # 13*13*208 = 35152, rf:17x17
        # 112 + 176 = 288
        network = swp_gadget(network, 96, stride=2, pad=1, name="goo4")

        # 5th.  Data size 13 -> 13
        # 13*13*128 = 21632, rf:33x33
        network = Conv2DLayer(network, 128, (3, 3), pad=1,
            W=HeNormal('relu'), name="conv5")
        network = apply_relu_bn(network)

        # 6th. Data size 13 -> 7
        # 7*7*256 = 12544, rf:49x49
        # 128 + 128 = 256 ch
        network = swp_gadget(network, 128, stride=2, pad=1, name="goo6")

        # 7th. Data size 7 -> 4
        # 4*4*448 = 7168, rf:81x81
        # 256 + 192 = 448 ch
        network = swp_gadget(network, 192, stride=2, pad=1, name="goo7")

        # 8th. Data size 4 -> 2
        # 2*2*1024 = 4096, rf:113x113
        # 448 + 576 = 1024 ch
        network = swp_gadget(network, 576, name="goo8")

        # 9th.
        network = Conv2DLayer(network, 2048, (2, 2), pad=0,
            W=HeNormal('relu'), name="conv9")
        network = apply_relu_bn(network)
        # Now enforce sparsity!
        network = FeatureWTALayer(network, 8)
        # network = MaxPool2DLayer(network, (2, 2))

        return network

    def regularize_parameters(self):
        l2 = lasagne.regularization.regularize_network_params(
            self.network, lasagne.regularization.l2) * 1e-2
        return l2

def swp_gadget(network_in, conv_add, stride=1, pad=0, wta=0, wtap=0, name=None):
    network_c = Conv2DLayer(network_in, conv_add // 2, (1, 1),
        W=HeNormal('relu'), name=name+'i')
    network_c = apply_relu_bn(network_c)
    conv = network_c = Conv2DLayer(network_c, conv_add, (3, 3),
        stride=stride, pad=pad, name=name,
        W=HeNormal('relu'))
    network_c = apply_relu_bn(network_c)
    if (wta > 1):
        network_c = FeatureWTALayer(network_c, wta)
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    if (wtap > 1):
        network_p = FeatureWTALayer(network_p, wtap)
    network_p = QuickNormLayer(network_p)
    return ConcatLayer((network_p, network_c), name=name + 'c')

def apply_relu_bn(layer):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn = layer.bn = QuickNormLayer(layer)
    out = layer.relu = NonlinearityLayer(bn, nonlinearity=rectify)
    return out

