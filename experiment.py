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

def apply_prelu_qn(layer):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn = layer.bn = QuickNormLayer(layer)
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

def rent(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 113 -> 111
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372), name="conv1")
    network = apply_prelu_bn_ren(network)
    # 2nd. Data size 111 -> 55
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = Conv2DLayer(network, 64, (3, 3), stride=2,
        W=HeNormal(1.372), name="conv2")

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

rent.cropsz = 113
rent.l2reg = 1e-2
rent.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
rent.l1reg = 1e-2
rent.l1map = {
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

def ww3(network, cropsz, batchsz):
    # 1st. Data size 99 -> 97
    network = Conv2DLayer(network, 64, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # 2nd. Data size 97 -> 95
    network = Conv2DLayer(network, 64, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # Max pool. Data size 95 -> 47
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd. Data size 47 -> 45
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # 4th. Data size 45 -> 43
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # Max pool. Data size 43 -> 21
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 5th. Data size 21 -> 19
    network = Conv2DLayer(network, 256, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # 6th. Data size 19 -> 17
    network = Conv2DLayer(network, 256, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # Max pool.  Data size 17 -> 8
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 7th. Data size 8 -> 6
    network = Conv2DLayer(network, 512, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # 8th. Data size 6 -> 4
    network = Conv2DLayer(network, 512, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # Max pool.  Data size 4 -> 2
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 9th. Data size 4 -> 1
    network = lasagne.layers.DenseLayer(network, 1024, W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    return network

ww3.cropsz = 99
ww3.l2reg = 1e-2
ww3.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)

def ww4(network, cropsz, batchsz):
    # 1st. Data size 94 -> 92
    network = Conv2DLayer(network, 64, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 2nd. Data size 90 -> 88
    network = Conv2DLayer(network, 64, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # Max pool. Data size 88 -> 44
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 3rd. Data size 44 -> 42
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 4th. Data size 42 -> 40
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # Max pool. Data size 40 -> 20
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 5th. Data size 20 -> 18
    network = Conv2DLayer(network, 256, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 6th. Data size 18 -> 16
    network = Conv2DLayer(network, 256, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # Max pool.  Data size 16 -> 8
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 7th. Data size 8 -> 6
    network = Conv2DLayer(network, 512, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 8th. Data size 6 -> 4
    network = Conv2DLayer(network, 512, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # Max pool.  Data size 4 -> 2
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 9th. Data size 4 -> 1
    network = lasagne.layers.DenseLayer(network, 1024, W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    return network

ww4.cropsz = 94
ww4.l2reg = 1e-2
ww4.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)

def ww5(network, cropsz, batchsz):
    # 1st. Data size 94 -> 92
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # 2nd. Data size 90 -> 88
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # Max pool. Data size 88 -> 44
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 3rd. Data size 44 -> 42
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # 4th. Data size 42 -> 40
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # Max pool. Data size 40 -> 20
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 5th. Data size 20 -> 18
    network = Conv2DLayer(network, 256, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # 6th. Data size 18 -> 16
    network = Conv2DLayer(network, 256, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # Max pool.  Data size 16 -> 8
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 7th. Data size 8 -> 6
    network = Conv2DLayer(network, 512, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # 8th. Data size 6 -> 4
    network = Conv2DLayer(network, 512, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # Max pool.  Data size 4 -> 2
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 9th. Data size 4 -> 1
    network = lasagne.layers.DenseLayer(network, 1024, W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    return network

ww5.cropsz = 94
ww5.l2reg = 1e-2
ww5.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)

def smart(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111
    network = Conv2DLayer(network, 64, (7, 7), stride=1,
        W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 55
    network = Conv2DLayer(network, 112, (5, 5), pad='same',
        W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 55 -> 27
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd.  Data size 27 -> 13
    network = Conv2DLayer(network, 192, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 27 -> 13
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 13 -> 11
    network = Conv2DLayer(network, 320, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 5th. Data size 5 -> 3
    network = Conv2DLayer(network, 512, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 6th. Data size 3 -> 1
    network = lasagne.layers.DenseLayer(network, 512,
        W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    network = DropoutLayer(network)
    # network = BatchNormLayer(network, nonlinearity=rectify)

    return network

def rena(network, cropsz, batchsz):
    # 1st. Data size 113 -> 111
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 55
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 55 -> 27
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3nd. Data size 27 -> 27
    network = Conv2DLayer(network, 96, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 27 -> 13
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 13 -> 13
    network = Conv2DLayer(network, 224, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 5th.  Data size 13 -> 11
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 6th. Data size 5 -> 5
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    return network

rena.cropsz = 113
rena.l2reg = 1e-2
rena.learning_rates = numpy.logspace(-1.5, -5, 30, dtype=numpy.float32)

def reda(network, cropsz, batchsz):
    # 1st. Data size 113 -> 111
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 55
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 55 -> 27
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3nd. Data size 27 -> 27
    network = Conv2DLayer(network, 96, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 27 -> 13
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 13 -> 13
    network = Conv2DLayer(network, 224, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 5th.  Data size 13 -> 11
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 6th. Data size 5 -> 5
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    network = DropoutLayer(network)

    return network

reda.cropsz = 113
reda.l2reg = 1e-2
reda.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)

#def srena(network, cropsz, batchsz):
#    network = ZeroGrayLayer(network)
#
#    # 1st. Data size 97 -> 95
#    # 113*113*32 = 408608, rf:3x3
#    network = Conv2DLayer(network, 32, (3, 3),
#        W=HeNormal('relu'), name="conv1")
#    network = apply_prelu_bn(network)
#
#    # 2nd. Data size 95 -> 47
#    # 55*55*64 = 193600, rf:5x5
#    # 32 + 32 = 64 ch
#    network = Conv2DLayer(network, 64, (3, 3), stride=2,
#        W=HeNormal('relu'), name="conv2")
#    network = apply_prelu_bn(network)
#
#    # 3nd. Data size 47 -> 23
#    # 27*27*96 = 69984, rf:9x9
#    # 64 + 32 = 96 ch
#    network = Conv2DLayer(network, 96, (3, 3), stride=2,
#        W=HeNormal('relu'), name="conv3")
#    network = apply_prelu_bn(network)
#
#    # 4th.  Data size 23 -> 11, 192 + 144
#    # 13*13*224 = 37856, rf:17x17
#    # 96 + 128 = 224 ch
#    network = Conv2DLayer(network, 224, (3, 3), stride=2,
#        W=HeNormal('relu'), name="conv4")
#    network = apply_prelu_bn(network)
#
#    # 4th.  Data size 11 -> 11
#    # 11*11*128 = 15488, rf:33x33
#    network = Conv2DLayer(network, 128, (3, 3), pad=1,
#        W=HeNormal(1.0), name="conv5")
#    network = apply_prelu_bn(network)
#
#    # 5th. Data size 11 -> 5
#    # 5*5*256 = 6400, rf:49x49
#    # 128 + 128 = 256 ch
#    network = Conv2DLayer(network, 256, (3, 3), stride=2,
#        W=HeNormal('relu'), name="conv6")
#    network = apply_prelu_bn(network)
#
#    # 6th. Data size 5 -> 3
#    # 3*3*384 = 3456, rf:81x81
#    # 128 + 256 = 384 ch
#    network = Conv2DLayer(network, 384, (3, 3),
#        W=HeNormal('relu'), name="conv7")
#    network = apply_prelu_bn(network)
#
#    # 7th. Data size 3 -> 1, 592 + 512 ch
#    # 1*1*896 = 896, rf:113x113
#    # 384 + 512 = 896 ch
#    network = Conv2DLayer(network, 896, (3, 3),
#        W=HeNormal('relu'), name="conv8")
#    network = apply_prelu_bn(network)
#
#    return network
#
#srena.cropsz = 97
#srena.l2reg = 1e-2
#srena.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)

def smar(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111, 394272
    network = Conv2DLayer(network, 32, (7, 7), stride=1,
        W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 55, 96800
    network = Conv2DLayer(network, 32, (5, 5), pad='same',
        W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 55 -> 27
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd.  Data size 27 -> 13, 27040
    network = Conv2DLayer(network, 160, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 27 -> 13
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 13 -> 11, 15488
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 5th. Data size 5 -> 3, 4608
    network = Conv2DLayer(network, 512, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 6th. Data size 3 -> 1, 1024
    network = lasagne.layers.DenseLayer(network, 1024,
        W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    network = DropoutLayer(network)
    # network = BatchNormLayer(network, nonlinearity=rectify)

    return network

def serena(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 99 -> 97
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 2nd. Data size 97 -> 95
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 95 -> 47
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd. Data size 47 -> 47
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 47 -> 23
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th. Data size 23 -> 23
    network = Conv2DLayer(network, 96, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 23 -> 11
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 5th.  Data size 11 -> 11
    network = Conv2DLayer(network, 224, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 6th.  Data size 13 -> 11
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 7th. Data size 5 -> 5
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 9th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    return network

serena.cropsz = 99
serena.l2reg = 1e-2
serena.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)

def qena(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 113 -> 111
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # max pool 111 -> 55
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 55
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # max pool 55 -> 27
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3nd. Data size 27 -> 27
    network = Conv2DLayer(network, 96, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # max pool 27 -> 13
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 13 -> 13
    network = Conv2DLayer(network, 224, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    # 5th.  Data size 13 -> 11
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 6th. Data size 5 -> 5
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_qn(network)

    return network

qena.cropsz = 113
# qena.l2reg = 1e-2
qena.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)

def renb(network, cropsz, batchsz):
    # 1st. Data size 113 -> 111
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 55
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 55 -> 27
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3nd. Data size 27 -> 27
    network = Conv2DLayer(network, 96, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 27 -> 13
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 13 -> 13
    network = Conv2DLayer(network, 224, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 5th.  Data size 13 -> 11
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    # network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 6th. Data size 11 -> 5
    network = Conv2DLayer(network, 256, (3, 3), stride=2, W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    return network

renb.cropsz = 113
# renb.l2reg = 1e-2
# renb.learning_rates = numpy.logspace(-1.5, -5, 30, dtype=numpy.float32)

def renc(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 113 -> 111
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    # network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 111 -> 55
    network = Conv2DLayer(network, 64, (3, 3), stride=2, W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 55 -> 27
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3nd. Data size 27 -> 27
    network = Conv2DLayer(network, 96, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 27 -> 13
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 13 -> 13
    network = Conv2DLayer(network, 224, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 5th.  Data size 13 -> 11
    network = Conv2DLayer(network, 128, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 6th. Data size 11 -> 5
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    return network

renc.cropsz = 113
# renc.l2reg = 1e-2
# renc.learning_rates = numpy.logspace(-1.5, -5, 30, dtype=numpy.float32)

def rend(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    # network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 95 -> 47
    network = Conv2DLayer(network, 64, (3, 3), stride=2, W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 47 -> 23
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3nd. Data size 23 -> 23
    network = Conv2DLayer(network, 96, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 23 -> 11
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 11 -> 11
    network = Conv2DLayer(network, 224, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 5th.  Data size 11 -> 11
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 6th. Data size 5 -> 5
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    return network

rend.cropsz = 113
# rend.l2reg = 1e-2
# rend.learning_rates = numpy.logspace(-1.5, -5, 30, dtype=numpy.float32)

def red(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    # network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 95 -> 47
    network = Conv2DLayer(network, 64, (3, 3), stride=2, W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 47 -> 23
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3nd. Data size 23 -> 23
    network = Conv2DLayer(network, 96, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 23 -> 11
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 11 -> 11
    network = Conv2DLayer(network, 224, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 5th.  Data size 11 -> 11
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 6th. Data size 5 -> 5
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = tprelu(network)
    network = DropoutLayer(network)
    # network = apply_prelu_bn(network)

    return network

red.cropsz = 113
red.ramp_lr = False
# red.l2reg = 1e-2
red.learning_rates = numpy.logspace(-1.5, -5, 30, dtype=numpy.float32)


def redd(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    # network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 95 -> 47
    network = Conv2DLayer(network, 64, (3, 3), stride=2, W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 47 -> 23
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3nd. Data size 23 -> 23
    network = Conv2DLayer(network, 96, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 23 -> 11
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 11 -> 11
    network = Conv2DLayer(network, 224, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)

    # 5th.  Data size 11 -> 11
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 6th. Data size 5 -> 5
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = tprelu(network)
    network = DropoutLayer(network)
    # network = apply_prelu_bn(network)

    return network

redd.cropsz = 113
redd.ramp_lr = False
# red.l2reg = 1e-2


def rudd(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 288800
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    # network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 141376
    # 2nd. Data size 95 -> 47
    network = Conv2DLayer(network, 64, (3, 3), stride=2, W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 47 -> 23
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 33856
    # 3nd. Data size 23 -> 23
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 23 -> 11
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 30976

    # 4th.  Data size 11 -> 11
    network = Conv2DLayer(network, 200, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 24200

    # 5th.  Data size 11 -> 11
    network = Conv2DLayer(network, 512, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 12800

    # 6th. Data size 5 -> 5
    # 6400
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = tprelu(network)
    network = DropoutLayer(network)
    # network = apply_prelu_bn(network)

    return network

rudd.cropsz = 113
rudd.ramp_lr = False
# red.l2reg = 1e-2




def rud(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 288800
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    # network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 141376
    # 2nd. Data size 95 -> 47
    network = Conv2DLayer(network, 128, (3, 3), stride=2, W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 47 -> 23
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 67712
    # 3nd. Data size 23 -> 23
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 23 -> 11
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 30976

    # 4th.  Data size 11 -> 11
    network = Conv2DLayer(network, 200, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 24200

    # 5th.  Data size 11 -> 11
    network = Conv2DLayer(network, 512, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 12800

    # 6th. Data size 5 -> 5
    # 6400
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = tprelu(network)
    network = DropoutLayer(network)
    # network = apply_prelu_bn(network)

    return network

rud.cropsz = 113
rud.ramp_lr = False
# red.l2reg = 1e-2

def rrd(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 288800
    network = Conv2DLayer(network, 32, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 111 -> 55
    # network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 2nd. Data size 95 -> 47
    # 141376
    network = Conv2DLayer(network, 64, (3, 3), stride=2, W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # conv. 47 -> 47
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 47 -> 23
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 67712
    # 3nd. Data size 23 -> 23
    network = Conv2DLayer(network, 320, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 23 -> 11
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 38720

    # 4th.  Data size 11 -> 11
    network = Conv2DLayer(network, 200, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 24200

    # 5th.  Data size 11 -> 11
    network = Conv2DLayer(network, 512, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # max pool 11 -> 5
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 12800

    # 6th. Data size 5 -> 5
    # 6400
    network = Conv2DLayer(network, 256, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 7th.  Data size 5 -> 3
    network = Conv2DLayer(network, 384, (3, 3), W=HeUniform('relu'))
    network = apply_prelu_bn(network)
    # 8th. Data size 3 -> 1
    network = Conv2DLayer(network, 896, (3, 3), W=HeUniform('relu'))
    network = tprelu(network)
    network = DropoutLayer(network)
    # network = apply_prelu_bn(network)

    return network

rrd.cropsz = 113
rrd.ramp_lr = False
# red.l2reg = 1e-2

