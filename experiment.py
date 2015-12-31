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

def gooey_gadget(network_in, conv_red, conv_add, stride=1, pad=0):
    network_c = Conv2DLayer(network_in, conv_red, (1, 1),
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_c = Conv2DLayer(network_c, conv_add, (3, 3), stride=stride, pad=pad,
        W=HeNormal(1.372))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride, pad=pad)
    return ConcatLayer((network_c, network_p))

def geefy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 115*115*32 = 438048, rf:3x3
    network = Conv2DLayer(network, 16, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 113*113*32 = 423200, rf:5x5
    network = Conv2DLayer(network, 16, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000, rf:7x7
    network = gooey_gadget(network, 16, 32, stride=2) # 32 + 32 = 64 ch

    # 2nd. Data size 55 -> 27
    # 27*27*96 = 69984, rf:11x11
    network = gooey_gadget(network, 16, 32, stride=2) # 64 + 32 = 96 ch

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856, rf:19x19
    network = gooey_gadget(network, 64, 128, stride=2) # 96 + 128 = 224 ch

    # 4th.  Data size 13 -> 11
    # 11*11*128 = 15488, rf:35x35
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:51x51
    network = gooey_gadget(network, 64, 128, stride=2) # 128 + 128 = 256 ch

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:83x83
    network = gooey_gadget(network, 64, 128) # 128 + 256 = 384 ch

    # 7th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*896 = 896, rf:115x115
    network = gooey_gadget(network, 256, 512) # 384 + 512 = 896 ch

    return network

geefy.cropsz = 115

# parameter count 1898820 (1895620 trainable) in 100 arrays
# Accuracy: 75.16% @5 after 30 epochs.
def gee(network, cropsz, batchsz):
    # 1st. Data size 113 -> 111
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 2nd. Data size 111 -> 55
    # 55*55*64 = 193600, rf:5x5
    network = gooey_gadget(network, 16, 32, stride=2) # 32 + 32 = 64 ch

    # 3nd. Data size 55 -> 27
    # 27*27*96 = 69984, rf:9x9
    network = gooey_gadget(network, 16, 32, stride=2) # 64 + 32 = 96 ch

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    network = gooey_gadget(network, 64, 128, stride=2) # 96 + 128 = 224 ch

    # 4th.  Data size 13 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    network = gooey_gadget(network, 64, 128, stride=2) # 128 + 128 = 256 ch

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    network = gooey_gadget(network, 64, 128) # 128 + 256 = 384 ch

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    network = gooey_gadget(network, 256, 512) # 384 + 512 = 896 ch

    return network

gee.cropsz = 113

def ree(network, cropsz, batchsz):
    # 1st. Data size 113 -> 111
    # 113*113*32 = 408608, rf:3x3
    network = Conv2DLayer(network, 32, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 2nd. Data size 111 -> 55
    # 55*55*64 = 193600, rf:5x5
    network = gooey_gadget(network, 16, 32, stride=2) # 32 + 32 = 64 ch

    # 3nd. Data size 55 -> 27
    # 27*27*96 = 69984, rf:9x9
    network = gooey_gadget(network, 16, 32, stride=2) # 64 + 32 = 96 ch

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*224 = 37856, rf:17x17
    network = gooey_gadget(network, 32, 128, stride=2) # 96 + 128 = 224 ch

    # 4th.  Data size 13 -> 11
    # 11*11*128 = 15488, rf:33x33
    network = Conv2DLayer(network, 128, (3, 3),
        W=HeNormal(1.372))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 5th. Data size 11 -> 5
    # 5*5*256 = 6400, rf:49x49
    network = gooey_gadget(network, 32, 128, stride=2) # 128 + 128 = 256 ch

    # 6th. Data size 5 -> 3
    # 3*3*384 = 3456, rf:81x81
    network = gooey_gadget(network, 32, 128) # 128 + 256 = 384 ch

    # 7th. Data size 3 -> 1, 592 + 512 ch
    # 1*1*896 = 896, rf:113x113
    network = gooey_gadget(network, 64, 512) # 384 + 512 = 896 ch

    return network

gee.cropsz = 113
