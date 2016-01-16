#!/usr/bin/env python3

from progressbar import ProgressBar
from response import ResponseProbe
from pretty import *
from model import Model
import argparse
import experiment
import lasagne
import theano
import theano.tensor as T
import pickle
import numpy
import time
import re
import random
import os
import os.path
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tagged', help='path to directory containing prepared files', default='tagged/full')
parser.add_argument('-n', '--network', help='name of network experiment', required=True)
parser.add_argument('-m', '--model', type=argparse.FileType('rb'), help='trained model to evaluate', required=True)
parser.add_argument('-b', '--batchsize', type=int, help='size of each mini batch', default=256)
parser.add_argument('-s', '--set', help='image set to evaluate on', choices=['test', 'val'], default='val')
parser.add_argument('-d', '--devkit', help='devkit directory containing categories.txt', default='mp-dev_kit')
parser.add_argument('-w', '--wrongskip', help='penalize images that are not correclty classified', type=bool, default=True)
parser.add_argument('-c', '--count', help='number of examples to output', type=int, default=50)
args = parser.parse_args()

imsz = 128
flips = [False, True]

section("Setup")
task("Loading data")
subtask("Loading categories")
cats = numpy.max(numpy.memmap(os.path.join(args.tagged, "train.labels.db"), dtype=numpy.int32, mode='r'))+1
subtask("Loading {} set".format(args.set))
y_test = numpy.memmap(os.path.join(args.tagged, "{}.labels.db".format(args.set)), dtype=numpy.int32, mode='r')
X_test = numpy.memmap(os.path.join(args.tagged, "{}.images.db".format(args.set)), dtype=numpy.float32, mode='r', shape=(len(y_test), 3, imsz, imsz))


task("Making output directory")
os.makedirs(os.path.join(os.path.dirname(args.model.name), 'ri'), exist_ok=True)

task("Building model and compiling functions")
model = Model(args.network, args.model.name)
debug_fn = model.debug_fn()
cropsz = model.cropsz

def iterate_minibatches(inputs):
    global args
    end = len(inputs)
    steps = range(0, end, args.batchsize)
    for start_idx in steps:
        yield inputs[slice(start_idx, start_idx + args.batchsize)]

section("Evaluation")
task("Evaluating results on {} data set".format(args.set))
cases = len(X_test)
layers = model.named_layers()
responses = {}
responselocs = {}
shapes = {}
for layer in layers:
  sh = lasagne.layers.get_output_shape(layer)
  shapes[layer.name] = sh
  responses[layer.name] = numpy.zeros((len(X_test), sh[1]))
  responselocs[layer.name] = numpy.zeros((len(X_test), sh[1]), dtype=numpy.int32)

test_batches = len(range(0, cases, args.batchsize))
p = progress(test_batches)
i = 0

# Collect max activation responses in each channel of each layer
# for every image in the test set.
for inp in iterate_minibatches(X_test):
    s = i * args.batchsize
    outs = debug_fn(inp)
    for j, layer in enumerate(layers):
        if len(outs[j].shape) == 4:
            sh = outs[j].shape
            flat = outs[j].reshape((sh[0], sh[1], sh[2] * sh[3]))
            responses[layer.name][s:s+len(inp),:] = numpy.max(flat, axis=2)
            responselocs[layer.name][s:s+len(inp),:] = numpy.argmax(flat, 2)
        else:
            responses[layer.name][s:s+len(inp),:] = outs[j]
            responselocs[layer.name][s:s+len(inp),:] = 0
    i += 1
    p.update(i)
p.finish()

if args.wrongskip:
    preds = responses['softmax'].argmax(axis=1)
    wrong = preds != y_test
    for layer in layers:
        responses[layer.name][wrong,:] -= 1

# Collect best 10 results for each channel of each layer
examples = {}
for layer in layers:
    best = (-responses[layer.name]).argsort(axis=0)[:args.count,:]
    examples[layer.name] = best

task("Generating response images".format(args.set))
i = 0
p = progress(args.count * sum([b.shape[1] for b in examples.values()]))

# Generate response image for each
for layer in reversed([lay in layers if lay.name == 'goo8c']):
    best = examples[layer.name]
    for j in range(best.shape[1]):
        for k in range(args.count):
            imdata = X_test[best[k, j]]
            if len(shapes[layer.name]) == 4:
                flatloc = responselocs[layer.name][best[k, j], j]
                loc = numpy.unravel_index(flatloc, shapes[layer.name][2:])
            else:
                loc = (0, 0)
            rp = ResponseProbe(model, imgdata=imdata)
            ri = rp.get_response_image(layer.name, j, loc[0], loc[1])
            imbytes = scipy.misc.bytescale(
                numpy.transpose(ri, [1, 2, 0]), cmin=0, cmax=1)
            outpath = os.path.join(os.path.dirname(args.model.name), 'ri',
                "{}_{}.{}.png".format(layer.name, j, k))
            scipy.misc.imsave(outpath, imbytes)
            i += 1
            p.update(i)
p.finish()
