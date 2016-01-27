#!/usr/bin/env python3

import pretty
import argparse
from periscope import prepare_corpus

parser = argparse.ArgumentParser()
parser.add_argument('--images', help='path to images/')
parser.add_argument('--devkit', help="path to dev kit's data/")
parser.add_argument('--corpus', help='path to directory in which to place serialized data files')
parser.add_argument('--width', help='pixel width', type=int, default=128)
parser.add_argument('--height', help='pixel height', type=int, default=128)
args = parser.parse_args()

prepare_corpus(args.devkit, args.images, args.outdir,
    width=args.width, height=args.height, seed=1, pretty=pretty)
