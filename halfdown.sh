#!/bin/sh

./makehalf.py --model=ww4bn2 \
   --reduce conv8 256 \
   --reduce conv7 256 \
   --reduce conv6 128 \
   --reduce conv5 128 \
   --reduce conv4 64 \
   --reduce conv3 64 \
   --reduce conv2 32 \
   --reduce conv1 32 \
   --save=ww4bn2-conv8-halfdown

./makehalf.py --model=ww4bn2 \
   --reduce conv7 256 \
   --reduce conv6 128 \
   --reduce conv5 128 \
   --reduce conv4 64 \
   --reduce conv3 64 \
   --reduce conv2 32 \
   --reduce conv1 32 \
   --save=ww4bn2-conv7-halfdown

./makehalf.py --model=ww4bn2 \
   --reduce conv6 128 \
   --reduce conv5 128 \
   --reduce conv4 64 \
   --reduce conv3 64 \
   --reduce conv2 32 \
   --reduce conv1 32 \
   --save=ww4bn2-conv6-halfdown

./makehalf.py --model=ww4bn2 \
   --reduce conv5 128 \
   --reduce conv4 64 \
   --reduce conv3 64 \
   --reduce conv2 32 \
   --reduce conv1 32 \
   --save=ww4bn2-conv5-halfdown

./makehalf.py --model=ww4bn2 \
   --reduce conv4 64 \
   --reduce conv3 64 \
   --reduce conv2 32 \
   --reduce conv1 32 \
   --save=ww4bn2-conv4-halfdown

./makehalf.py --model=ww4bn2 \
   --reduce conv3 64 \
   --reduce conv2 32 \
   --reduce conv1 32 \
   --save=ww4bn2-conv3-halfdown

./makehalf.py --model=ww4bn2 \
   --reduce conv2 32 \
   --reduce conv1 32 \
   --save=ww4bn2-conv2-halfdown

./eval.py --model ww4bn2-conv8-halfdown --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv7-halfdown --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv6-halfdown --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv5-halfdown --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv4-halfdown --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv3-halfdown --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv2-halfdown --net ww4bn2 --truncate
