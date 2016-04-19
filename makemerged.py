#!/usr/bin/env python3

import pretty
import argparse
from periscope import Corpus, Checkpoint
from periscope.debug import ActivationSample
from periscope.debug import NetworkMerger
from periscope import prepare_corpus, load_from_checkpoint, class_for_shortname

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', help='corpus directory', default='corpus/mp')
parser.add_argument('--net1', help='name of network model class name',
     default=None)
parser.add_argument('--model1', help='directory for checkpoints', default=None)
parser.add_argument('--net2', help='name of network model class name',
     default=None)
parser.add_argument('--model2', help='directory for checkpoints', default=None)
parser.add_argument('--save', help='directory to save reduced model')
args = parser.parse_args()

# Usage: makemerge --model1 ww4bn --model2 ww4bn-2
#   This will add the two networks right before their output layer.
def load_net_for(modelname, netname):
    # default model name is the net name.
    if modelname is None:
        modelname = netname.split('.')[-1].lower()
    # models go into the model directory unless a directory is specified.
    if '/' not in modelname:
        modelname = 'model/' + modelname
    if netname is not None:
        # use an explicit net name.
        net = class_for_shortname(netname)(model=modelname)
    else:
        try:
            # an existing model specifies its network class.
            net = load_from_checkpoint(modelname)
        except:
            # a network class name can be guessed by using the model name.
            shortname = modelname.split('/')[-1].split('-')[0]
            net = class_for_shortname(shortname)(model=modelname)
    return net

if '/' not in args.save:
    args.save = 'model/' + args.save

net1 = load_net_for(args.model1, args.net1)
net2 = load_net_for(args.model2, args.net2)

# Load the corpus
corpus = Corpus(args.corpus)

# After the training, generate purpose database and images.
# sample1 = ActivationSample(net1, corpus, pretty=pretty)
# sample2 = ActivationSample(net2, corpus, pretty=pretty)

# After the training, generate purpose database and images.
merger = NetworkMerger([net1, net2], corpus)
merged = merger.create_merged_network()
merged.checkpoint = Checkpoint(args.save)
merged.save_checkpoint()

