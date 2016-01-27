#!/usr/bin/env python3

import pretty
import argparse
from periscope import Corpus
from periscope import prepare_corpus, load_from_checkpoint, class_for_shortname

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', help='corpus directory', default='corpus/mp')
parser.add_argument('--network', help='name of network model class name',
     default=None)
parser.add_argument('--model', help='directory for checkpoints', default=None)
args = parser.parse_args()

assert args.model or args.network

# Load the network and associate it with the model directory.
if args.model is None:
    args.model = args.network.split('.')[-1].lower()
if '/' not in args.model:
    args.model = 'model/' + args.model

if args.network is not None:
    net = class_for_shortname(args.network)(model=args.model)
else:
    net = load_from_checkpoint(args.model)

# Load the corpus
corpus = Corpus(args.corpus)

# Do the training
net.train(corpus, pretty=pretty)
