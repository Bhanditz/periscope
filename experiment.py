import lasagne
import theano
import numpy
from lasagne.init import HeUniform, HeNormal
from lasagne.layers import Layer
from lasagne.layers import DropoutLayer, FeaturePoolLayer, ConcatLayer, prelu
from lasagne.layers import DenseLayer, ReshapeLayer, SliceLayer, FeatureWTALayer
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

def apply_qn_only(layer):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn = layer.bn = QuickNormLayer(layer)
    return bn

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
    conv.prelu.name = name + 'r'
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    network_p = QuickNormLayer(network_p)
    return ConcatLayer((network_p, network_c), name=name + 'c')

# parameter count 1895940 (1890820 trainable)
# achieves about 48%@1 and 77%@5 on miniplaces challenge.
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
    conv.prelu.name = name + 'r'
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

def ww4(network, cropsz, batchsz):
    # 1st. Data size 96->96
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))
    # 2nd. Data size 96->96
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))

    # Max pool. Data size 96->48
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 3rd. Data size 48->48
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))
    # 4th. Data size 48->48
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))

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

    return network

ww4.cropsz = 96
ww4.batchsize = 64
ww4.l2reg = 0
ww4.ramp_lr = False
ww4.learning_rates = numpy.concatenate((
  [0.01] * 13,
  [0.001] * 2,
  [0.0001] * 15
))

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



# parameter count 7245132 (7241596 trainable)
# Achieves about 44.2%@1 and 74.7%@5
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

def qren(network, cropsz, batchsz):
    return ren(network, cropsz, batchsz)

qren.cropsz = 113
qren.batchsize = 64
qren.l2reg = 1e-2
qren.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
qren.l1reg = 1e-2
qren.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def apply_relu_bn_cen(layer):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn = layer.bn = QuickNormLayer(layer)
    out = layer.prelu = NonlinearityLayer(bn)
    out = ZeroReluLayer(out)
    return out

class ZeroReluLayer(Layer):
    def __init__(self, incoming, alpha=0, **kwargs):
        super(ZeroReluLayer, self).__init__(incoming, **kwargs)
        self.mean = numpy.float32(numpy.sqrt(2/numpy.pi)*(1 -alpha)/2)
        self.invstd = numpy.float32(1/numpy.sqrt(0.5*((1-2/numpy.pi)*(1 + alpha) +
                      1/numpy.pi * (1 + alpha)**2)))
    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_output_for(self, input, **kwargs):
        return (input - self.mean) * self.invstd

def cen_gadget(network_in, conv_add, stride=1, pad=0, name=None):
    network_c = Conv2DLayer(network_in, conv_add // 2, (1, 1),
        W=HeNormal(1.372), name=name+'i')
    network_c = apply_relu_bn_cen(network_c)
    conv = network_c = Conv2DLayer(network_c, conv_add, (3, 3),
        stride=stride, pad=pad, name=name,
        W=HeNormal(1.372))
    network_c = apply_relu_bn_cen(network_c)
    conv.bn.name = name + 'b'
    conv.prelu.name = name + 'r'
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    network_p = QuickNormLayer(network_p)
    return ConcatLayer((network_p, network_c), name=name + 'c')

# Just like ren, but using regular RELU instead of prelu.
# parameter count 1895940 (1890820 trainable)
# achieves about 48%@1 and 77%@5 on miniplaces challenge.
def cen(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 113 -> 111
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372), name="conv1")
    network = apply_relu_bn_cen(network)
    # 2nd. Data size 111 -> 55
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = cen_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 55 -> 27
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = cen_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = cen_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 13 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372), name="conv5")
    network = apply_relu_bn_cen(network)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = cen_gadget(network, 128, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = cen_gadget(network, 128, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = cen_gadget(network, 512, name="goo8")

    return network

cen.cropsz = 113
cen.l2reg = 1e-2
cen.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
cen.l1reg = 1e-2
cen.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def wta_gadget(network_in, conv_add, stride=1, pad=0, wta=0, name=None):
    network_c = Conv2DLayer(network_in, conv_add // 2, (1, 1),
        W=HeNormal(1.372), name=name+'i')
    network_c = apply_prelu_bn_ren(network_c)
    conv = network_c = Conv2DLayer(network_c, conv_add, (3, 3),
        stride=stride, pad=pad, name=name,
        W=HeNormal(1.372))
    network_c = apply_prelu_bn_ren(network_c)
    if (wta > 1):
        network_c = FeatureWTALayer(network_c, wta)
    conv.bn.name = name + 'b'
    conv.prelu.name = name + 'r'
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    network_p = QuickNormLayer(network_p)
    return ConcatLayer((network_p, network_c), name=name + 'c')

# parameter count 1895940 (1890820 trainable)
# achieves about 48%@1 and 77%@5 on miniplaces challenge.
def wta(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 113 -> 111
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372), name="conv1")
    network = apply_prelu_bn_ren(network)
    # 2nd. Data size 111 -> 55
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = wta_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 55 -> 27
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = wta_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = wta_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 13 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372), name="conv5")
    network = apply_prelu_bn_ren(network)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = wta_gadget(network, 128, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = wta_gadget(network, 128, name="goo7", wta=8)

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = wta_gadget(network, 512, name="goo8", wta=16)

    return network

wta.cropsz = 113
wta.l2reg = 1e-2
wta.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
wta.l1reg = 1e-2
wta.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def bnww(network, cropsz, batchsz):
    # 1st. Data size 96->96
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn_sren(network)

    # 2nd. Data size 96->96
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn_sren(network)

    # Max pool. Data size 96->48
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 3rd. Data size 48->48
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn_sren(network)

    # 4th. Data size 48->48
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))
    network = apply_prelu_bn_sren(network)

    # Max pool. Data size 48->24
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 5th. Data size 24->24
    network = Conv2DLayer(network, 256, (3, 3), pad='same',
        W=HeUniform('relu'), name='conv5')
    network = apply_prelu_bn_sren(network)

    # 6th. Data size 24->24
    network = Conv2DLayer(network, 256, (3, 3), pad='same',
        W=HeUniform('relu'), name='conv6')
    network = apply_prelu_bn_sren(network)

    # Max pool.  Data size 24->12
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 7th. Data size 12->12
    network = Conv2DLayer(network, 512, (3, 3), pad='same',
        W=HeUniform('relu'), name='conv7')
    network = apply_prelu_bn_sren(network)

    # 8th. Data size 12->12
    network = Conv2DLayer(network, 512, (3, 3), pad='same',
        W=HeUniform('relu'), name='conv8')
    network = apply_prelu_bn_sren(network)

    # Max pool.  Data size 12->6
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 9th. Data size 6->1
    network = lasagne.layers.DenseLayer(network, 1024, W=HeUniform('relu'))
    network = apply_prelu_bn_sren(network)

    return network

bnww.cropsz = 96
bnww.batchsize = 64
bnww.l2reg = 5e-3
bnww.ramp_lr = True
bnww.learning_rates = numpy.concatenate((
  [0.01] * 13,
  [0.001] * 13,
  [0.0001] * 13,
  [0.00001] * 13
))

def swa_gadget(network_in, conv_add, stride=1, pad=0, wta=0, wtap=0, name=None):
    network_c = Conv2DLayer(network_in, conv_add // 2, (1, 1),
        W=HeNormal(1.372), name=name+'i')
    network_c = apply_prelu_bn_sren(network_c)
    conv = network_c = Conv2DLayer(network_c, conv_add, (3, 3),
        stride=stride, pad=pad, name=name,
        W=HeNormal(1.372))
    network_c = apply_prelu_bn_sren(network_c)
    if (wta > 1):
        network_c = FeatureWTALayer(network_c, wta)
    conv.bn.name = name + 'b'
    conv.prelu.name = name + 'r'
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    if (wtap > 1):
        network_p = FeatureWTALayer(network_p, wtap)
    network_p = QuickNormLayer(network_p)
    return ConcatLayer((network_p, network_c), name=name + 'c')

def swa(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.0), name="conv1")
    network = apply_prelu_bn_sren(network)
    # 2nd. Data size 95 -> 47
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = swa_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 47 -> 23
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = swa_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 23 -> 11, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = swa_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 11 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 128, (3, 3), pad=1,
        W=HeNormal(1.0), name="conv5")
    network = apply_prelu_bn_sren(network)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = swa_gadget(network, 128, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = swa_gadget(network, 128, wta=16, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = swa_gadget(network, 512, wta=16, name="goo8")

    return network

swa.cropsz = 97
swa.batchsize = 256
swa.l2reg = 1e-2
swa.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
swa.l1reg = 1e-2
swa.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def sww(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.0), name="conv1")
    network = apply_prelu_bn_sren(network)
    # 2nd. Data size 95 -> 47
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = swa_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 47 -> 23
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = swa_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 23 -> 11, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = swa_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 11 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 128, (3, 3), pad=1,
        W=HeNormal(1.0), name="conv5")
    network = apply_prelu_bn_sren(network)
    network = FeatureWTALayer(network, 2)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = swa_gadget(network, 128, wta=4, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = swa_gadget(network, 128, wta=4, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = swa_gadget(network, 512, wta=8, name="goo8")

    return network

sww.cropsz = 97
sww.batchsize = 256
sww.l2reg = 1e-2
sww.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
sww.l1reg = 1e-2
sww.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def swx(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.0), name="conv1")
    network = apply_prelu_bn_sren(network)
    # 2nd. Data size 95 -> 47
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = swa_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 47 -> 23
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = swa_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 23 -> 11, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = swa_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 11 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 180, (3, 3), pad=1,
        W=HeNormal(1.0), name="conv5")
    network = apply_prelu_bn_sren(network)
    network = FeatureWTALayer(network, 2)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = swa_gadget(network, 256, wta=4, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = swa_gadget(network, 256, wta=4, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = swa_gadget(network, 1024, wta=8, name="goo8")

    return network

swx.cropsz = 97
swx.batchsize = 256
swx.l2reg = 1e-2
swx.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
swx.l1reg = 1e-2
swx.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def swy(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.0), name="conv1")
    network = apply_prelu_bn_sren(network)
    # 2nd. Data size 95 -> 47
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = swa_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 47 -> 23
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = swa_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 23 -> 11, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = swa_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 11 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 256, (3, 3), pad=1,
        W=HeNormal(1.0), name="conv5")
    network = apply_prelu_bn_sren(network)
    network = FeatureWTALayer(network, 4)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = swa_gadget(network, 256, wta=4, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = swa_gadget(network, 256, wta=4, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = swa_gadget(network, 1024, wta=8, name="goo8")

    return network

swy.cropsz = 97
swy.batchsize = 256
swy.l2reg = 1e-2
swy.learning_rates = numpy.logspace(-2, -5.33333, 40, dtype=numpy.float32)
swy.l1reg = 1e-2
swy.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def swz(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.0), name="conv1")
    network = apply_prelu_bn_sren(network)
    # 2nd. Data size 95 -> 47
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = swa_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 47 -> 23
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = swa_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 23 -> 11, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = swa_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 11 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 256, (3, 3), pad=1,
        W=HeNormal(1.0), name="conv5")
    network = apply_prelu_bn_sren(network)
    network = FeatureWTALayer(network, 4)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = swa_gadget(network, 256, wta=4, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = swa_gadget(network, 256, wta=4, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = swa_gadget(network, 1024, wta=4, name="goo8")

    return network

swz.cropsz = 97
swz.batchsize = 256
swz.l2reg = 1e-2
swz.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
swz.l1reg = 1e-2
swz.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def swh(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.0), name="conv1")
    network = apply_prelu_bn_sren(network)
    # 2nd. Data size 95 -> 47
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = swa_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 47 -> 23
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = swa_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 23 -> 11, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = swa_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 11 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 256, (3, 3), pad=1,
        W=HeNormal(1.0), name="conv5")
    network = apply_prelu_bn_sren(network)
    network = FeatureWTALayer(network, 4)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = swa_gadget(network, 256, wta=4, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = swa_gadget(network, 256, wta=4, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = swa_gadget(network, 1020, wta=6, name="goo8")

    return network

swh.cropsz = 97
swh.batchsize = 256
swh.l2reg = 1e-2
swh.learning_rates = numpy.logspace(-2, -4.5, 30, dtype=numpy.float32)
swh.l1reg = 1e-2
swh.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def sfy_gadget(network_in, conv_add, stride=1, pad=0, wta=0, name=None):
    network_c = Conv2DLayer(network_in, conv_add // 2, (1, 1),
        W=HeNormal(1.372), name=name+'i')
    network_c = apply_prelu_bn_sren(network_c)
    conv = network_c = Conv2DLayer(network_c, conv_add, (3, 3),
        stride=stride, pad=pad, name=name,
        W=HeNormal(1.372))
    if (wta > 1):
        network_c = apply_qn_only(network_c)
        network_c = FeatureWTALayer(network_c, wta)
    else:
        network_c = apply_prelu_bn_sren(network_c)
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    network_p = QuickNormLayer(network_p)
    return ConcatLayer((network_p, network_c), name=name + 'c')

def sfy(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.0), name="conv1")
    network = apply_prelu_bn_sren(network)
    # 2nd. Data size 95 -> 47
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = sfy_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 47 -> 23
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = sfy_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 23 -> 11, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = sfy_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 11 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 256, (3, 3), pad=1,
        W=HeNormal(1.0), name="conv5")
    network = apply_prelu_bn_sren(network)
    network = FeatureWTALayer(network, 4)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = sfy_gadget(network, 256, wta=4, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = sfy_gadget(network, 256, wta=4, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = sfy_gadget(network, 1024, wta=8, name="goo8")

    return network

sfy.cropsz = 97
sfy.batchsize = 256
sfy.l2reg = 1e-2
sfy.learning_rates = numpy.logspace(-2, -5.33333, 40, dtype=numpy.float32)
sfy.l1reg = 1e-2
sfy.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def sfe(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.0), name="conv1")
    network = apply_prelu_bn_sren(network)
    # 2nd. Data size 95 -> 47
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = sfy_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 47 -> 23
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = sfy_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 23 -> 11, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = sfy_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 11 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 256, (3, 3), pad=1,
        W=HeNormal(1.0), name="conv5")
    network = apply_prelu_bn_sren(network)
    network = FeatureWTALayer(network, 8)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = sfy_gadget(network, 256, wta=8, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = sfy_gadget(network, 256, wta=8, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = sfy_gadget(network, 1024, wta=8, name="goo8")

    return network

sfe.cropsz = 97
sfe.batchsize = 256
sfe.l2reg = 1e-2
sfe.learning_rates = numpy.logspace(-2, -5.33333, 40, dtype=numpy.float32)
sfe.l1reg = 1e-2
sfe.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def sey(network, cropsz, batchsz):
    network = ZeroGrayLayer(network)

    # 1st. Data size 97 -> 95
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.0), name="conv1")
    network = apply_prelu_bn_sren(network)
    # 2nd. Data size 95 -> 47
    # 55*55*64 = 193600, rf:5x5
    # 32 + 32 = 64 ch
    network = swa_gadget(network, 32, stride=2, name="goo2")

    # 3nd. Data size 47 -> 23
    # 27*27*96 = 69984, rf:9x9
    # 64 + 32 = 96 ch
    network = swa_gadget(network, 32, stride=2, name="goo3")

    # 3rd.  Data size 23 -> 11, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    # 96 + 128 = 224 ch
    network = swa_gadget(network, 128, stride=2, name="goo4")

    # 4th.  Data size 11 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 256, (3, 3), pad=1,
        W=HeNormal(1.0), name="conv5")
    network = apply_prelu_bn_sren(network)
    # network = FeatureWTALayer(network, 4)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    # 128 + 128 = 256 ch
    network = swa_gadget(network, 256, stride=2, name="goo6")

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    # 128 + 256 = 384 ch
    network = swa_gadget(network, 256, name="goo7")

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    # 384 + 512 = 896 ch
    network = swa_gadget(network, 1024, name="goo8")

    return network

sey.cropsz = 97
sey.batchsize = 256
sey.l2reg = 1e-2
sey.learning_rates = numpy.logspace(-2, -5.33333, 40, dtype=numpy.float32)
sey.l1reg = 1e-2
sey.l1map = {
    'conv5': 1,
    'goo6i': 1,
    'goo6': 1,
    'goo7i': 1,
    'goo7': 1,
    'goo8i': 1,
    'goo8': 1,
    'softmax': 1
}

def ww3wd(network, cropsz, batchsz):
    # 1st. Data size 96->96
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))
    # 2nd. Data size 96->96
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))

    # Max pool. Data size 96->48
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 3rd. Data size 48->48
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))
    # 4th. Data size 48->48
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))

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

    # Max pool.  Data size 12->3
    network = MaxPool2DLayer(network, (4, 4), stride=4)

    # 9th. Data size 3->1
    network = lasagne.layers.DenseLayer(network, 1024, W=HeUniform('relu'))

    return network

ww3wd.cropsz = 96
ww3wd.batchsize = 64
ww3wd.l2reg = 5e-3
ww3wd.ramp_lr = False
ww3wd.learning_rates = numpy.concatenate((
  [0.01] * 20,
  [0.001] * 20,
  [0.0001] * 20
))

def ww4wd(network, cropsz, batchsz):
    # 1st. Data size 96->96
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))
    # 2nd. Data size 96->96
    network = Conv2DLayer(network, 64, (3, 3), pad='same', W=HeUniform('relu'))

    # Max pool. Data size 96->48
    network = MaxPool2DLayer(network, (2, 2), stride=2)

    # 3rd. Data size 48->48
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))
    # 4th. Data size 48->48
    network = Conv2DLayer(network, 128, (3, 3), pad='same', W=HeUniform('relu'))

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

    return network

ww4wd.cropsz = 96
ww4wd.batchsize = 64
ww4wd.l2reg = 5e-3
ww4wd.ramp_lr = False
ww4wd.learning_rates = numpy.concatenate((
  [0.01] * 32,
  [0.001] * 32,
  [0.0001] * 32
))

