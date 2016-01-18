import lasagne
import theano
import numpy
from lasagne.init import HeUniform, HeNormal
from lasagne.layers import Layer
from lasagne.layers import DropoutLayer, FeaturePoolLayer, ConcatLayer, prelu
from lasagne.layers import DenseLayer, ReshapeLayer, SliceLayer
from lasagne.layers import ParametricRectifierLayer, NonlinearityLayer
from lasagne.layers import get_output_shape
from lasagne.layers.normalization import BatchNormLayer
from lasagne.nonlinearities import rectify, softmax, identity, LeakyRectify
Conv2DLayer = lasagne.layers.Conv2DLayer
MaxPool2DLayer = lasagne.layers.MaxPool2DLayer
Pool2DLayer = lasagne.layers.Pool2DLayer
if theano.config.device.startswith("gpu"):
    import lasagne.layers.dnn
    # Force GPU implementations if a GPU is available.
    # Do not know why Theano is not selecting these impls
    # by default as advertised.
    if theano.sandbox.cuda.dnn.dnn_available():
        Conv2DLayer = lasagne.layers.dnn.Conv2DDNNLayer
        MaxPool2DLayer = lasagne.layers.dnn.MaxPool2DDNNLayer
        Pool2DLayer = lasagne.layers.dnn.Pool2DDNNLayer

def tprelu(network):
    out = prelu(network)
    network.prelu = out
    return out

def apply_prelu_bn(layer):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn = layer.bn = BatchNormLayer(layer)
    out = layer.prelu = ParametricRectifierLayer(bn)
    return out

def tprelu_bn(network):
    out = prelu(network)
    network.prelu = out
    return out

class ZeroGrayLayer(Layer):
    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_output_for(self, input, **kwargs):
        return input - 0.5

def base(network, cropsz, batchsz):
    # 1st
    network = Conv2DLayer(network, 64, (8, 8), stride=2, nonlinearity=rectify)
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 2nd
    network = Conv2DLayer(network, 96, (5, 5), stride=1, pad='same')
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 3rd
    network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same')
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 4th
    network = lasagne.layers.DenseLayer(network, 512)
    network = BatchNormLayer(network, nonlinearity=rectify)

    return network

def smarter(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    network = Conv2DLayer(network, 64, (7, 7), stride=1,
        W=HeUniform('relu'))
    network = tprelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 27
    network = Conv2DLayer(network, 112, (5, 5), stride=1, pad='same',
        W=HeUniform('relu'))
    network = tprelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd.  Data size 27 -> 13
    network = Conv2DLayer(network, 192, (3, 3), stride=1, pad='same',
        W=HeUniform('relu'))
    network = tprelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 11 -> 5
    network = Conv2DLayer(network, 320, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = tprelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 5th. Data size 5 -> 3
    network = Conv2DLayer(network, 512, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = tprelu(network)
    network = BatchNormLayer(network)

    # 6th. Data size 3 -> 1
    network = lasagne.layers.DenseLayer(network, 512,
        W=HeUniform('relu'))
    network = tprelu(network)
    network = DropoutLayer(network)
    # network = BatchNormLayer(network, nonlinearity=rectify)

    return network

class ZeroPreluLayer(Layer):
    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_output_for(self, input, **kwargs):
        return (input - 0.2) * 1.25

class QuickNormLayer(Layer):
    def __init__(self, incoming,
                 axes='auto', epsilon=1e-4, alpha=0.1,
                 mode='low_mem',
                 mean=lasagne.init.Constant(0),
                 inv_std=lasagne.init.Constant(1), **kwargs):
        super(QuickNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha
        self.mode = mode

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("QuickNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        self.mean = self.add_param(inv_std, shape, 'mean',
                                      trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = theano.tensor.inv(theano.tensor.sqrt(
                input.var(self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        use_averages = kwargs.get('batch_norm_use_averages',
                                  deterministic)
        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        update_averages = kwargs.get('batch_norm_update_averages',
                                     not deterministic)
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * inv_std
        return normalized

def apply_prelu_bn_ren(layer):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn = layer.bn = QuickNormLayer(layer)
    out = layer.prelu = ParametricRectifierLayer(bn)
    out = ZeroPreluLayer(out)
    return out

def ren_gadget(network_in, conv_add, stride=1, pad=0, name=None):
    network_c = Conv2DLayer(network_in, conv_add // 2, (1, 1),
        W=HeNormal(1.372), name=name+'i')
    network_c = apply_prelu_bn_ren(network_c)
    conv = network_c = Conv2DLayer(network_c, conv_add, (3, 3),
        stride=stride, pad=pad, name=name,
        W=HeNormal(1.372))
    network_c = apply_prelu_bn_ren(network_c)
    conv.bn.name = name + 'b'
    conv.prelu.name = name + 'p'
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    network_p = QuickNormLayer(network_p)
    return ConcatLayer((network_p, network_c), name=name + 'c')

def ren(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 113 -> 111
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372), name="conv1")
    network = apply_prelu_bn_ren(network)
    # 2nd. Data size 111 -> 55
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = ren_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 55 -> 27
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = ren_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = ren_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 13 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372), name="conv5")
    network = apply_prelu_bn_ren(network)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = ren_gadget(network, 128, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = ren_gadget(network, 128, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = ren_gadget(network, 512, name="goo8")

    return network

ren.cropsz = 113
ren.l2reg = 1e-2
ren.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
ren.l1reg = 1e-2
ren.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

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
    conv.prelu.name = name + 'p'
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    network_p = QuickNormLayer(network_p)
    return ConcatLayer((network_p, network_c), name=name + 'c')

def sren(network, cropsz, batchsz):
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

sren.cropsz = 97
sren.l2reg = 1e-2
sren.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
sren.l1reg = 1e-2
sren.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}
