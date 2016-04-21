#!/usr/bin/env python3

import pretty
import argparse
from periscope import Corpus, Checkpoint
from periscope.debug import NetworkPermuter
from periscope import load_from_checkpoint, class_for_shortname

parser = argparse.ArgumentParser()
parser.add_argument('--net', help='name of network model class name',
     default=None)
parser.add_argument('--model', help='directory for checkpoints', default=None)
parser.add_argument('--perm', help='layer name and size',
     nargs='+', action='append')
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

# Decode permutation arguments
permutations = []
for p in args.perm:
    layername = p[0]
    perm = []
    for j in range(1, len(p), 2):
        perm.extend(range(int(p[j]), int(p[j + 1])))
    permutations.append((layername, perm))

# After the training, generate purpose database and images.
reducer = NetworkPermuter(net)
reduced = reducer.create_permuted_network(permutations)
reduced.checkpoint = Checkpoint(args.save)
reduced.save_checkpoint()
