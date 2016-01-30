from periscope import Network
from periscope.layers import ZeroGrayLayer, QuickNormLayer, ZeroPreluLayer
from periscope.layers import LandmarkLayer
import lasagne
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, ConcatLayer
from lasagne.layers import ParametricRectifierLayer
from lasagne.init import HeNormal
from lasagne.nonlinearities import identity
import numpy as np

def apply_prelu_bn_sren(layer):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn = layer.bn = QuickNormLayer(layer)
    out = layer.prelu = ParametricRectifierLayer(bn)
    return out

def sren_gadget(network_in, conv_add, stride=1, pad=0, name=None):
    network_c = Conv2DLayer(network_in, conv_add // 2, (1, 1),
        W=HeNormal(1.0), name=name+'i')
    network_c = apply_prelu_bn_sren(network_c)
    conv = network_c = Conv2DLayer(network_c, conv_add, (3, 3),
        stride=stride, pad=pad, name=name,
        W=HeNormal(1.0))
    network_c = apply_prelu_bn_sren(network_c)
    conv.bn.name = name + 'b'
    conv.prelu.name = name + 'r'
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    network_p = QuickNormLayer(network_p)
    return ConcatLayer((network_p, network_c), name=name + 'c')


# 1895940 params
class Srel(Network):

    def init_constants(self):
        super(Srel, self).init_constants()
        self.crop_size = (97, 97)
        self.learning_rates = np.logspace(-2, -4.5, 30, dtype=np.float32)

    def hidden_layers(self, network, **kwargs):
        network = ZeroGrayLayer(network)

        # 1st. Data size 97 -> 95
        # 113*113*32 = 408608, rf:3x3
        network = Conv2DLayer(network, 32, (3, 3),
            W=HeNormal(1.0), name="conv1")
        network = apply_prelu_bn_sren(network)
        # 2nd. Data size 95 -> 47
        # 55*55*64 = 193600, rf:5x5
        # 32 + 32 = 64 ch
        network = sren_gadget(network, 32, stride=2, name="goo2")

        # 3nd. Data size 47 -> 23
        # 27*27*96 = 69984, rf:9x9
        # 64 + 32 = 96 ch
        network = sren_gadget(network, 32, stride=2, name="goo3")

        # 3rd.  Data size 23 -> 11, 192 + 144
        # 13*13*224 = 37856, rf:17x17
        # 96 + 128 = 224 ch
        network = sren_gadget(network, 128, stride=2, name="goo4")

        # 4th.  Data size 11 -> 11
        # 11*11*128 = 15488, rf:33x33
        network = Conv2DLayer(network, 128, (3, 3), pad=1,
            W=HeNormal(1.0), name="conv5")
        network = apply_prelu_bn_sren(network)

        # Add landmarks!
        network = LandmarkLayer(network)

        # 5th. Data size 11 -> 5
        # 5*5*256 = 6400, rf:49x49
        # 128 + 128 = 256 ch
        network = sren_gadget(network, 128, stride=2, name="goo6")

        # 6th. Data size 5 -> 3
        # 3*3*384 = 3456, rf:81x81
        # 128 + 256 = 384 ch
        network = sren_gadget(network, 128, name="goo7")

        # 7th. Data size 3 -> 1, 592 + 512 ch
        # 1*1*896 = 896, rf:113x113
        # 384 + 512 = 896 ch
        network = sren_gadget(network, 512, name="goo8")

        return network

    def regularize_parameters(self):
        l2 = lasagne.regularization.regularize_network_params(
            self.network, lasagne.regularization.l2) * 1e-2
        return l2

