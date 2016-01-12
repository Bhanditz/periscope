#!/usr/bin/env python3

from progressbar import ProgressBar
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

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tagged', help='path to directory containing prepared files', default='tagged/full')
parser.add_argument('-n', '--network', help='name of network experiment', nargs='+', required=True)
parser.add_argument('-m', '--model', type=argparse.FileType('rb'), help='trained model to evaluate', nargs='+', required=True)
parser.add_argument('-b', '--batchsize', type=int, help='size of each mini batch', default=256)
parser.add_argument('-s', '--set', help='image set to evaluate on', choices=['test', 'val'], default='test')
parser.add_argument('-l', '--labels', action='store_true', help='output category labels', default=False)
parser.add_argument('-d', '--devkit', help='devkit directory containing categories.txt', default='mp-dev_kit')
args = parser.parse_args()

assert len(args.network) == len(args.model)

imsz = 128
flips = [False, True]

section("Setup")
task("Loading data")
subtask("Loading categories")
cats = numpy.max(numpy.memmap(os.path.join(args.tagged, "train.labels.db"), dtype=numpy.int32, mode='r'))+1
subtask("Loading {} set".format(args.set))
y_test = numpy.memmap(os.path.join(args.tagged, "{}.labels.db".format(args.set)), dtype=numpy.int32, mode='r')
X_test = numpy.memmap(os.path.join(args.tagged, "{}.images.db".format(args.set)), dtype=numpy.float32, mode='r', shape=(len(y_test), 3, imsz, imsz))

section("Constructing networks")
networks = []
cropsz = []
for ni, m in enumerate(args.model):
    task(m.name)
    subtask("Building model and compiling functions")
    model = Model(args.network[ni], m.name)
    networks.append(model.eval_fn())
    cropsz.append(model.cropsz)

def iterate_minibatches(inputs):
    global args
    end = len(inputs)
    steps = range(0, end, args.batchsize)
    for start_idx in steps:
        yield inputs[slice(start_idx, start_idx + args.batchsize)]

section("Evaluation")
task("Evaluating performance on {} data set".format(args.set))
cases = len(X_test)
predictions = numpy.zeros((len(X_test), 5))

test_batches = len(range(0, cases, args.batchsize))
p = progress(test_batches)
i = 0
_preds = numpy.zeros((2*3*3*len(networks), args.batchsize, cats))

for inp in iterate_minibatches(X_test):
    s = i * args.batchsize
    if s + args.batchsize > predictions.shape[0]:
        inp = inp[:predictions.shape[0] - s]

    config = 0
    _preds.fill(0)
    for ni, network in enumerate(networks):
        for flip in [-1, 1]:
            sz = cropsz[ni]
            edge = imsz - sz
            center = edge // 2
            for xcrop in [0, center, edge]:
                for ycrop in [0, center, edge]:
                    _preds[config, :len(inp), :] = network(
                            inp, flip, (ycrop, xcrop))
                    config += 1

    # take median across configurations
    # pick top 5 categories
    # last category is highest probability
    avgscore = numpy.median(_preds, axis=0)
    predictions[s:s+len(inp), :] = numpy.argsort(avgscore)[:len(inp), -5:][:, ::-1]
    i += 1
    p.update(i)

filenames = [line.strip() for line in open(os.path.join(args.tagged,
        'test.filenames.txt')).readlines()]

categories = {}
if args.labels:
    with open(os.path.join(args.devkit, "categories.txt"), 'r') as cmap:
        for line in cmap:
            c, ci = line.split(None, 1)
            categories[int(ci)] = os.path.basename(c)

for i in range(len(predictions)):
    if args.labels:
        cats = "\t".join([categories[ci] for ci in predictions[i]])
    else:
        cats = " ".join([str(int(c)) for c in predictions[i]])
    print("{} {}".format(filenames[i], cats))

if args.set != 'test':
    top1 = numpy.mean(numpy.equal(predictions[:, 0], y_test))
    top5 = numpy.mean(numpy.any(numpy.equal(predictions[:, 0:5], y_test.reshape(-1, 1)), axis=1))
    task("Evaluation accuracy: exact: {}, top-5: {}".format(top1, top5))
