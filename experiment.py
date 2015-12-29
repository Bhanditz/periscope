import lasagne
import theano
import numpy
from lasagne.init import HeUniform, HeNormal
from lasagne.layers import DropoutLayer, FeaturePoolLayer, ConcatLayer, prelu
from lasagne.layers import DenseLayer, ReshapeLayer, SliceLayer
from lasagne.layers import get_output_shape
from lasagne.layers.normalization import BatchNormLayer
from lasagne.nonlinearities import rectify, softmax
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

# This one achieves about 27.5% err@5
def deeper(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    network = Conv2DLayer(network, 64, (7, 7), stride=1)
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 27
    network = Conv2DLayer(network, 112, (5, 5), stride=1, pad='same')
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd.  Data size 27 -> 13
    network = Conv2DLayer(network, 192, (3, 3), stride=1, pad='same')
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 11 -> 5
    network = Conv2DLayer(network, 320, (3, 3), stride=1)
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 5th. Data size 5 -> 3
    network = Conv2DLayer(network, 512, (3, 3), stride=1)
    # network = DropoutLayer(network)
    network = BatchNormLayer(network, nonlinearity=rectify)

    # 6th. Data size 3 -> 1
    network = lasagne.layers.DenseLayer(network, 512)
    network = DropoutLayer(network)
    # network = BatchNormLayer(network, nonlinearity=rectify)

    return network


def slim(network, cropsz, batchsz):
    # 1st
    network = Conv2DLayer(network, 64, (5, 5), stride=2, W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (5, 5), stride=2)
    # 2nd
    network = Conv2DLayer(network, 96, (5, 5), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (5, 5), stride=2)
    # 3rd
    network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 4th
    network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 5th
    network = lasagne.layers.DenseLayer(network, 512, W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)

    return network

def cslim(network, cropsz, batchsz):
    # 1st
    network = Conv2DLayer(network, 64, (5, 5), stride=2, W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (5, 5), stride=2)
    # 2nd
    network = Conv2DLayer(network, 96, (5, 5), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (5, 5), stride=2)
    # 3rd
    network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 4th
    network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = DropoutLayer(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 5th
    network = lasagne.layers.DenseLayer(network, 512, nonlinearity=None)
    network = DropoutLayer(network)
    network = FeaturePoolLayer(network, 2)

    return network

def smarter(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    network = Conv2DLayer(network, 64, (7, 7), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 27
    network = Conv2DLayer(network, 112, (5, 5), stride=1, pad='same',
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd.  Data size 27 -> 13
    network = Conv2DLayer(network, 192, (3, 3), stride=1, pad='same',
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 11 -> 5
    network = Conv2DLayer(network, 320, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 5th. Data size 5 -> 3
    network = Conv2DLayer(network, 512, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 6th. Data size 3 -> 1
    network = lasagne.layers.DenseLayer(network, 512,
        W=HeUniform('relu'))
    network = prelu(network)
    network = DropoutLayer(network)
    # network = BatchNormLayer(network, nonlinearity=rectify)

    return network

def choosy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    network = Conv2DLayer(network, 64, (7, 7), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 27
    network = Conv2DLayer(network, 112, (5, 5), stride=1, pad='same',
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd.  Data size 27 -> 13
    network = Conv2DLayer(network, 192, (3, 3), stride=1, pad='same',
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 11 -> 5
    network = Conv2DLayer(network, 320, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 5th. Data size 5 -> 3
    network = Conv2DLayer(network, 512, (3, 3), nonlinearity=None)
    network = prelu(network)
    network = BatchNormLayer(network)

    # 6th. Data size 3 -> 1
    network = lasagne.layers.DenseLayer(network, 512, nonlinearity=None)
    network = DropoutLayer(network)
    network = FeaturePoolLayer(network, 2)

    return network

def gooey_gadget(network_in, conv_add, stride=1, pad=0):
    network_c = Conv2DLayer(network_in, conv_add / 2, (1, 1),
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_c = Conv2DLayer(network_c, conv_add, (3, 3), stride=stride, pad=pad,
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    return ConcatLayer((network_c, network_p))

# parameter count 2601172 (2597028 trainable) in 93 arrays
def gooey(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = Conv2DLayer(network, 40, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = Conv2DLayer(network, 96, (3, 3), stride=2,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 92 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 192, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network


# parameter count 2601172 (2597028 trainable) in 93 arrays
def goopy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 40, 2) # 32 + 40 = 72 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = Conv2DLayer(network, 96, (3, 3), stride=2,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 92 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 192, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network


def goofy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 192, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goory(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels
    shortcut = MaxPool2DLayer(network, (3, 3), stride=1)

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 192, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = ConcatLayer((network, shortcut))  # Another 224 channels
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goosy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels
    shortcut = MaxPool2DLayer(network, (7, 7), stride=3)

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 192, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels
    network = ConcatLayer((network, shortcut))  # Another 224 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def gooty(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels
    shortcut = MaxPool2DLayer(network, (5, 5), stride=2)

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 192, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels
    network = ConcatLayer((network, shortcut))  # Another 224 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goouy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels
    shortcut = MaxPool2DLayer(network, (13, 13))

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 192, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels
    network = ConcatLayer((network, shortcut))  # Another 224 channels

    return network

def googy_gadget(network_in, conv_add, stride):
    network_c = Conv2DLayer(network_in, conv_add / 2, (1, 1),
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_c = Conv2DLayer(network_c, conv_add, (3, 3), stride=stride,
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride)
    network_p = Conv2DLayer(network_p,
        get_output_shape(network_in)[1] / 2, (1, 1),
        W=HeNormal(1.372))
    network_p = prelu(network_c)
    network_p = BatchNormLayer(network_p)
    return ConcatLayer((network_c, network_p))

def googy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = googy_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = googy_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = googy_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 192, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = googy_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = googy_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = googy_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goozy_gadget(network_in, conv_add, stride):
    network_c = Conv2DLayer(network_in, conv_add / 2, (1, 1),
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_c = Conv2DLayer(network_c, conv_add, (3, 3), stride=stride,
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    return network_c

def goozy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = goozy_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = goozy_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = goozy_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 192, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = goozy_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = goozy_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = goozy_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofx(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 96, (1, 1),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    network = Conv2DLayer(network, 192, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy256(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 256, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy384(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 384, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy64(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 64, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goody128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = DropoutLayer(network)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goody256(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 256, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = DropoutLayer(network)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goodx128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 64, (1, 1),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = DropoutLayer(network)

    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def foofy128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*354 = 8800
    network = gooey_gadget(network, 224, 2) # 128 + 224 = 354 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 5202
    network = gooey_gadget(network, 224, 1) # 354 + 224 = 578 channels

    # 6th. Data size 3 -> 1, 578 + 1536 channels
    # 1*1*2114 = 2114
    network = gooey_gadget(network, 512, 1) # 578 + 1536 = 2114 channels

    return network

def foofy128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*354 = 8800
    network = gooey_gadget(network, 224, 2) # 128 + 224 = 352 channels

    # 5th. Data size 5 -> 3
    # 3*3*608 = 5472
    network = gooey_gadget(network, 256, 1) # 352 + 256 = 608 channels

    # 6th. Data size 3 -> 1, 608 + 1536 channels
    # 1*1*1184 = 1632
    network = gooey_gadget(network, 1024, 1) # 608 + 1024 = 1632 channels

    # 7th. Data size 1 -> 1, 512 channels
    # 1*1*512 = 512
    network = lasagne.layers.DenseLayer(network, 512)
    network = prelu(network)
    network = BatchNormLayer(network)
    return network

def oldDoofyPyramid(network, index, channels):
    net = SliceLayer(network, index, axis=2)
    for depth in channels:
        net = lasagne.layers.DenseLayer(net, depth)
        net = prelu(net)
        net = BatchNormLayer(net)
    return net

def doofyPyramid(net, channels):
    for depth in channels:
        net = lasagne.layers.DenseLayer(net, depth)
        net = prelu(net)
        net = BatchNormLayer(net)
    return net

def olddoofy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*354 = 8800
    network = gooey_gadget(network, 224, 2) # 128 + 224 = 352 channels

    # 5th. Data size 5 -> 3
    # 3*3*608 = 5472
    network = gooey_gadget(network, 256, 1) # 352 + 256 = 608 channels

    # 6th. Data size 3 -> 1, 608 + 1536 channels
    # 4*1*1*762 = 3048
    network = ReshapeLayer(network, (-1, 152, 4, 3, 3))
    net0 = SliceLayer(network, 0, axis=2)
    net1 = SliceLayer(network, 1, axis=2)
    net2 = SliceLayer(network, 2, axis=2)
    net3 = SliceLayer(network, 3, axis=2)
    net0 = gooey_gadget(net0, 600, 1) # 152+600 = 762 channels
    net1 = gooey_gadget(net1, 600, 1) # 152+600 = 762 channels
    net2 = gooey_gadget(net2, 600, 1) # 152+600 = 762 channels
    net3 = gooey_gadget(net3, 600, 1) # 152+600 = 762 channels

    # 7th. Data size 1 -> 1, 1696 channels
    # 8*376 = 1696
    net0 = ReshapeLayer(net0, (-1, 376, 2))
    net1 = ReshapeLayer(net1, (-1, 376, 2))
    net2 = ReshapeLayer(net2, (-1, 376, 2))
    net3 = ReshapeLayer(net3, (-1, 376, 2))
    pyramid = (208, ) # , 116, 64, 36, 20, 11)
    net4 = oldDoofyPyramid(net0, 1, pyramid)
    net5 = oldDoofyPyramid(net1, 1, pyramid)
    net6 = oldDoofyPyramid(net2, 1, pyramid)
    net7 = oldDoofyPyramid(net3, 1, pyramid)
    net0 = oldDoofyPyramid(net0, 0, pyramid)
    net1 = oldDoofyPyramid(net1, 0, pyramid)
    net2 = oldDoofyPyramid(net2, 0, pyramid)
    net3 = oldDoofyPyramid(net3, 0, pyramid)
    network = ConcatLayer((net0, net1, net2, net3, net4, net5, net6, net7))
    return network

def doofy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*354 = 8800
    network = gooey_gadget(network, 224, 2) # 128 + 224 = 352 channels

    # 5th. Data size 5 -> 3
    # 3*3*608 = 5472
    network = gooey_gadget(network, 256, 1) # 352 + 256 = 608 channels

    # 6th. Data size 3 -> 1
    # 608 + 2400 = 3008
    slices = 8
    nets = [None] * slices
    network = ReshapeLayer(network, (-1, 608 // slices, slices, 3, 3))
    for s in range(slices):
        net = SliceLayer(network, s, axis=2)
        nets[s] = gooey_gadget(net, 2400 // slices, 1) # 38+150 = 188 channels

    # 7th.  Pyramid down: 1664 -> 928 -> 512
    subs = 2
    pyramid = (x // (slices * subs) for x in [1664, 928, 512])
    pyrs = [None] * slices * subs
    for s in range(slices):
        network = ReshapeLayer(nets[s], (-1, 3008 // slices // subs, subs))
        for t in range(subs):
           net = SliceLayer(network, t, axis=2)
           pyrs[s * subs + t] = doofyPyramid(net, pyramid)
    network = ConcatLayer(pyrs)

    # 8th.  Last layer
    network = doofyPyramid(network, (288, 160))
    return network

def hoofy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    # 7th and 8th.
    network = doofyPyramid(network, (288, ))

    return network

def hoofy640(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    # 7th and 8th.
    network = doofyPyramid(network, (640, ))

    return network

def hoofy640d(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    # 7th and 8th.
    network = doofyPyramid(network, (640, ))
    network = DropoutLayer(network)

    return network

def hoofy640c(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels
    net0 = ReshapeLayer(network, (-1, 1120))

    # 7th and 8th.
    network = doofyPyramid(network, (640, ))
    network = ConcatLayer((network, net0))

    return network

def hoofy160(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    # 7th and 8th.
    network = doofyPyramid(network, (160, ))

    return network

def goofy12_128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy24_128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 24, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy24d128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 24, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 24, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy12d128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy12t128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*12 = 164268
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 158700
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*32 = 96800
    network = gooey_gadget(network, 20, 2) # 12 + 20 = 32 channels

    # 2nd. Data size 55 -> 27
    # 27*27*72 = 52488
    network = gooey_gadget(network, 40, 2) # 40 + 32 = 72 channels

    # 3rd.  Data size 27 -> 13
    # 13*13*200 = 33800
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 200 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy3s12t128(network, cropsz, batchsz):
    # 0th layer: 3 combinations of RGB.
    net0 = Conv2DLayer(network, 3, (1, 1),
        W=HeNormal(1.372))
    net0 = prelu(net0)
    net0 = BatchNormLayer(net0)
    network = ConcatLayer((network, net0))

    # 1st. Data size 117 -> 111 -> 55
    # 117*117*12 = 164268
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 158700
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*32 = 96800
    network = gooey_gadget(network, 20, 2) # 12 + 20 = 32 channels

    # 2nd. Data size 55 -> 27
    # 27*27*72 = 52488
    network = gooey_gadget(network, 40, 2) # 40 + 32 = 72 channels

    # 3rd.  Data size 27 -> 13
    # 13*13*200 = 33800
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 200 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

class TileLayer(lasagne.layers.Layer):
    def __init__(self, incoming, repeats, **kwargs):
        super(TileLayer, self).__init__(incoming, **kwargs)
        self.repeats = repeats

    def get_output_shape_for(self, input_shape):
        return tuple(x * y for x, y in zip(input_shape, self.repeats))

    def get_output_for(self, input, **kwargs):
        return theano.tensor.tile(input, self.repeats)

def goofy3f12t128(network, cropsz, batchsz):
    # 0th layer: 3 combinations of RGB.
    net0 = Pool2DLayer(network, 3, (117, 117))
    net0 = TileLayer(net0, (1, 1, 117, 117))
    network = ConcatLayer((network, net0))

    # 1st. Data size 117 -> 111 -> 55
    # 117*117*12 = 164268
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 158700
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*32 = 96800
    network = gooey_gadget(network, 20, 2) # 12 + 20 = 32 channels

    # 2nd. Data size 55 -> 27
    # 27*27*72 = 52488
    network = gooey_gadget(network, 40, 2) # 40 + 32 = 72 channels

    # 3rd.  Data size 27 -> 13
    # 13*13*200 = 33800
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 200 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network


class NormValueLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(NormValueLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        mean = theano.tensor.mean(input, (2, 3), keepdims=True, dtype='float32')
        var = theano.tensor.var(input, (1, 2, 3), keepdims=True)
        return (input - mean) / var

def goofy3m12t128(network, cropsz, batchsz):
    # 0th layer: 3 combinations of RGB.
    net0 = NormValueLayer(network)
    network = ConcatLayer((network, net0))

    # 1st. Data size 117 -> 111 -> 55
    # 117*117*12 = 164268
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 158700
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*32 = 96800
    network = gooey_gadget(network, 20, 2) # 12 + 20 = 32 channels

    # 2nd. Data size 55 -> 27
    # 27*27*72 = 52488
    network = gooey_gadget(network, 40, 2) # 40 + 32 = 72 channels

    # 3rd.  Data size 27 -> 13
    # 13*13*200 = 33800
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 200 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

class NormGrayLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(NormGrayLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = 1
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        r = input[:,0:1,:,:]
        g = input[:,1:2,:,:]
        b = input[:,2:3,:,:]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        mean = theano.tensor.mean(y, (2, 3), keepdims=True, dtype='float32')
        var = theano.tensor.var(y, (2, 3), keepdims=True)
        return (y - mean) / var

def goofy3g12t128(network, cropsz, batchsz):
    # 0th layer: add a normalized gray layer.
    net0 = NormGrayLayer(network)
    network = ConcatLayer((network, net0))

    # 1st. Data size 117 -> 111 -> 55
    # 117*117*12 = 164268
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 158700
    network = Conv2DLayer(network, 12, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*32 = 96800
    network = gooey_gadget(network, 20, 2) # 12 + 20 = 32 channels

    # 2nd. Data size 55 -> 27
    # 27*27*72 = 52488
    network = gooey_gadget(network, 40, 2) # 40 + 32 = 72 channels

    # 3rd.  Data size 27 -> 13
    # 13*13*200 = 33800
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 200 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy28d128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 28, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 28, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy36d128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 36, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 36, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def goofy36t128(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 36, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 36, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 36, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 36, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = gooey_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = gooey_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def geefy256(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*384 = 9600
    network = gooey_gadget(network, 256, 2) # 128 + 256 = 384 channels

    # 5th. Data size 5 -> 3
    # 3*3*640 = 5670
    network = gooey_gadget(network, 256, 1) # 384 + 256 = 640 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def geefy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, stride=2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, stride=2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, stride=2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5*5*256 = 6400
    network = gooey_gadget(network, 128, stride=2) # 128 + 128 = 256 channels

    # 5th. Data size 5 -> 3
    # 3*3*384 = 3456
    network = gooey_gadget(network, 128) # 128 + 256 = 384 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512) # 672 + 512 = 1184 channels

    return network

def geefy2(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, stride=2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, stride=2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, stride=2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 11*11*256, Extra thinking here.    
    network = gooey_gadget(network, 128, pad=1)

    # 5*5*256 = 6400
    network = gooey_gadget(network, 128, stride=2) # 128 + 128 = 256 channels

    # 5th. Data size 5 -> 3
    # 3*3*384 = 3456
    network = gooey_gadget(network, 128) # 128 + 256 = 384 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512) # 672 + 512 = 1184 channels

    return network

def geefy22(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, stride=2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, stride=2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, stride=2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 9*9*256, Extra thinking here.    
    network = gooey_gadget(network, 128)

    # 7*7*256, Extra thinking here.    
    network = gooey_gadget(network, 128)

    # 5*5*256 = 6400
    network = gooey_gadget(network, 128) # 128 + 128 = 256 channels

    # 5th. Data size 5 -> 3
    # 3*3*384 = 3456
    network = gooey_gadget(network, 128) # 128 + 256 = 384 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512) # 672 + 512 = 1184 channels

    return network

def geefy23(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, stride=2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, stride=2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, stride=2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 9*9*256, Extra thinking here.    
    network = gooey_gadget(network, 32)

    # 7*7*256, Extra thinking here.    
    network = gooey_gadget(network, 32)

    # 5*5*256 = 6400
    network = gooey_gadget(network, 128) # 128 + 128 = 256 channels

    # 5th. Data size 5 -> 3
    # 3*3*384 = 3456
    network = gooey_gadget(network, 128) # 128 + 256 = 384 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512) # 672 + 512 = 1184 channels

    return network

def geefy3(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, stride=2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, stride=2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, stride=2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5*5*256 = 6400
    network = gooey_gadget(network, 128, stride=2) # 128 + 128 = 256 channels

    # 5th. Data size 5 -> 3
    # 3*3*384 = 3456
    network = gooey_gadget(network, 128) # 128 + 256 = 384 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512) # 672 + 512 = 1184 channels

    # 7th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = DenseLayer(network, 512,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    return network

def geefy4(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = gooey_gadget(network, 32, stride=2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = gooey_gadget(network, 32, stride=2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = gooey_gadget(network, 128, stride=2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*128 = 15488
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    net0 = network = BatchNormLayer(network)

    # 11*11*256, Extra thinking here.    
    network = Conv2DLayer(network, 128, (3, 3), pad=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = ConcatLayer((net0, network))

    # 5*5*384 = 9600
    network = gooey_gadget(network, 128, stride=2) # 128 + 256 = 384 channels

    # 5th. Data size 5 -> 3
    # 3*3*384 = 4608
    network = gooey_gadget(network, 128) # 128 + 384 = 512 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = gooey_gadget(network, 512) # 512 + 512 = 1024 channels

    return network


def groovy_gadget(network_in, conv_add, stride=1, pad=0):
    network_c = Conv2DLayer(network_in, conv_add / 2, (1, 1),
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_c = Conv2DLayer(network_c, conv_add, (3, 3), stride=stride, pad=pad,
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_pm = SliceLayer(network_in, slice(0, None, 2), axis=1)
    network_pm = MaxPool2DLayer(network_pm, (3, 3), stride=stride, pad=pad)
    network_pa = SliceLayer(network_in, slice(1, None, 2), axis=1)
    network_pa = Pool2DLayer(network_in, (3, 3), stride=stride, pad=pad,
        mode='average_exc_pad')
    return ConcatLayer((network_c, network_pm, network_pa))

def groovy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = groovy_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = groovy_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = groovy_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = groovy_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = groovy_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = groovy_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

def groopy_gadget(network_in, conv_add, stride=1, pad=0):
    network_c = Conv2DLayer(network_in, conv_add / 2, (1, 1),
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_c = Conv2DLayer(network_c, conv_add, (3, 3), stride=stride, pad=pad,
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    halfdepth = get_output_shape(network_in)[1] * 2 // 3
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    network_p = Conv2DLayer(network_p, halfdepth, (1, 1),
        W=HeNormal(1.372))
    return ConcatLayer((network_c, network_p))

def groopy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = groopy_gadget(network, 32, 2) # 32 + 32 = 64 channels

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984
    network = groopy_gadget(network, 32, 2) # 64 + 32 = 96 channels

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856
    network = groopy_gadget(network, 128, 2) # 96 + 128 = 224 channels

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*192 = 23232
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # network = DropoutLayer(network, p=0.25)

    # 5*5*412 = 10400
    network = groopy_gadget(network, 224, 2) # 192 + 224 = 416 channels

    # 5th. Data size 5 -> 3
    # 3*3*672 = 6048
    network = groopy_gadget(network, 256, 1) # 416 + 256 = 672 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1184 = 1184
    network = groopy_gadget(network, 512, 1) # 672 + 512 = 1184 channels

    return network

