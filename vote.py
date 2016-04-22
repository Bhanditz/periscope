#!/usr/bin/env python3

import pretty
import argparse
import numpy as np
from periscope import Corpus
from periscope.debug import PurposeMapper
from periscope import prepare_corpus, load_from_checkpoint, class_for_shortname

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', help='corpus directory', default='corpus/mp')
parser.add_argument('--net', help='names of network model class name', nargs='+')
parser.add_argument('--model', help='directories for checkpoints', nargs='+')
parser.add_argument('--epoch', help='starting epoch', type=int, default=None)
parser.add_argument('--truncate', dest='truncate', action='store_true')
parser.add_argument('--no-truncate', dest='truncate', action='store_false')
parser.set_defaults(truncate=False)

args = parser.parse_args()

def load_net(modelname, netname):
    # default model name is the net name.
    if modelname is None:
        modelname = netname.split('.')[-1].lower()
    # models go into the model directory unless a directory is specified.
    if '/' not in modelname:
        modelname = 'model/' + modelname

    if netname is not None:
        # use an explicit net name.
        net = class_for_shortname(netname)(
            model=modelname, truncate=args.truncate, epoch=args.epoch)
    else:
        try:
            # an existing model specifies its network class.
            net = load_from_checkpoint(modelname,
                    truncate=args.truncate, epoch=args.epoch)
        except:
            # a network class name can be guessed by using the model name.
            shortname = modelname.split('/')[-1].split('-')[0]
            net = class_for_shortname(shortname)(model=modelname)
    return net

# Load the corpus
corpus = Corpus(args.corpus)

# Collect the consensus votes (just via the sum)
netname = None
consensus = None
for index, modelname in enumerate(args.model):
    if args.net and index < len(args.net):
        netname = args.net[index]
    net = load_net(modelname, netname)
    pred = net.predict(corpus, pretty=pretty)
    if consensus is None:
        consensus = pred
    else:
        consensus += pred

preds = np.argsort(-consensus, axis=1)
valset = corpus.batches('val')
labels = np.expand_dims(valset.raw_y(), 1)
ranks = abs(preds - labels).argmin(axis=1)
c_1 = sum(ranks == 0)
c_5 = sum(ranks < 5)
d = len(ranks)
result = (c_1 / float(d), c_5 / float(d))
print("Accuracy: %.2f@1, %.2f@5" % (result[0] * 100, result[1] * 100))
