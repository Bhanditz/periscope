import lasagne
import theano
from lasagne.init import HeUniform, HeNormal
from lasagne.layers import DropoutLayer, FeaturePoolLayer, ConcatLayer, prelu
from lasagne.layers import get_output_shape
from lasagne.layers.normalization import BatchNormLayer
from lasagne.nonlinearities import rectify, softmax
Conv2DLayer = lasagne.layers.Conv2DLayer
MaxPool2DLayer = lasagne.layers.MaxPool2DLayer
if theano.config.device.startswith("gpu"):
    import lasagne.layers.dnn
    # Force GPU implementations if a GPU is available.
    # Do not know why Theano is not selecting these impls
    # by default as advertised.
    if theano.sandbox.cuda.dnn.dnn_available():
        Conv2DLayer = lasagne.layers.dnn.Conv2DDNNLayer
        MaxPool2DLayer = lasagne.layers.dnn.MaxPool2DDNNLayer

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

def gooey_gadget(network_in, conv_add, stride):
    network_c = Conv2DLayer(network_in, conv_add / 2, (1, 1),
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_c = Conv2DLayer(network_c, conv_add, (3, 3), stride=stride,
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride)
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

