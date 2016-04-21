#!/usr/bin/env python3

import pretty
import argparse
from periscope import Corpus, Checkpoint
from periscope.debug import NetworkReducer
from periscope import prepare_corpus, load_from_checkpoint, class_for_shortname

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', help='corpus directory', default='corpus/mp')
parser.add_argument('--net', help='name of network model class name',
     default=None)
parser.add_argument('--model', help='directory for checkpoints', default=None)
parser.add_argument('--reduce', help='layer name and size',
     nargs=2, action='append')
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

# After the training, generate purpose database and images.
reducer = NetworkReducer(net, corpus)
reduced = reducer.create_reduced_network(args.reduce)
reduced.checkpoint = Checkpoint(args.save)
reduced.save_checkpoint()
