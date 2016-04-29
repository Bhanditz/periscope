#!/usr/bin/env python3

import pretty
import argparse
from periscope import Corpus, Checkpoint
from periscope.debug import ActivationSample, NetworkOrderer, NetworkPermuter
from periscope import prepare_corpus, load_from_checkpoint, class_for_shortname

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', help='corpus directory', default='corpus/mp')
parser.add_argument('--net', help='name of network model class name',
     default=None)
parser.add_argument('--model', help='directory for checkpoints', default=None)
parser.add_argument('--layer', help='layer names', nargs=2, action='append')
parser.add_argument('--lowlayer', help='layers to discard lowest first', nargs=2, action='append')
parser.add_argument('--save', help='directory to save reduced model')
args = parser.parse_args()

assert args.model or args.net

# default model name is the net name.
if args.model is None:
    args.model = args.net.split('.')[-1].lower()
# models go into the model directory unless a directory is specified.
if '/' not in args.model:
    args.model = 'model/' + args.model

if args.net is not None:
    # use an explicit net name.
    net = class_for_shortname(args.net)(model=args.model, truncate=True)
else:
    try:
        # an existing model specifies its network class.
        net = load_from_checkpoint(args.model, truncate=True)
    except:
        # a network class name can be guessed by using the model name.
        shortname = args.model.split('/')[-1].split('-')[0]
        net = class_for_shortname(shortname)(model=args.model)

if '/' not in args.save:
    args.save = 'model/' + args.save

# Load the corpus
corpus = Corpus(args.corpus)

# Ensure the activation sample is created.
sample = ActivationSample(net, corpus, force=False, pretty=pretty)

# Create an orderer
orderer = NetworkOrderer(net, sample)
loworder = orderer.order_units_by_high_residuals(args.lowlayer)
order = orderer.order_units_by_high_residuals(args.layer)
print(loworder + order)
permuter = NetworkPermuter(net)
permuted = permuter.create_permuted_network(loworder + order)
permuted.checkpoint = Checkpoint(args.save)
permuted.save_checkpoint()
