#!/bin/sh

./makehalf.py --model=ww4bn2 \
   --reduce fc9 512 \
   --reduce conv8 256 \
   --save=ww4bn2-conv8-halfup

./makehalf.py --model=ww4bn2 \
   --reduce fc9 512 \
   --reduce conv8 256 \
   --reduce conv7 256 \
   --save=ww4bn2-conv7-halfup

./makehalf.py --model=ww4bn2 \
   --reduce fc9 512 \
   --reduce conv8 256 \
   --reduce conv7 256 \
   --reduce conv6 128 \
   --save=ww4bn2-conv6-halfup

./makehalf.py --model=ww4bn2 \
   --reduce fc9 512 \
   --reduce conv8 256 \
   --reduce conv7 256 \
   --reduce conv6 128 \
   --reduce conv5 128 \
   --save=ww4bn2-conv5-halfup

./makehalf.py --model=ww4bn2 \
   --reduce fc9 512 \
   --reduce conv8 256 \
   --reduce conv7 256 \
   --reduce conv6 128 \
   --reduce conv5 128 \
   --reduce conv4 64 \
   --save=ww4bn2-conv4-halfup

./makehalf.py --model=ww4bn2 \
   --reduce fc9 512 \
   --reduce conv8 256 \
   --reduce conv7 256 \
   --reduce conv6 128 \
   --reduce conv5 128 \
   --reduce conv4 64 \
   --reduce conv3 64 \
   --save=ww4bn2-conv3-halfup

./makehalf.py --model=ww4bn2 \
   --reduce fc9 512 \
   --reduce conv8 256 \
   --reduce conv7 256 \
   --reduce conv6 128 \
   --reduce conv5 128 \
   --reduce conv4 64 \
   --reduce conv3 64 \
   --reduce conv2 32 \
   --save=ww4bn2-conv2-halfup

./makehalf.py --model=ww4bn2 \
   --reduce fc9 512 \
   --reduce conv8 256 \
   --reduce conv7 256 \
   --reduce conv6 128 \
   --reduce conv5 128 \
   --reduce conv4 64 \
   --reduce conv3 64 \
   --reduce conv2 32 \
   --reduce conv1 32 \
   --save=ww4bn2-conv1-halfup

./eval.py --model ww4bn2-conv8-halfup --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv7-halfup --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv6-halfup --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv5-halfup --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv4-halfup --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv3-halfup --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv2-halfup --net ww4bn2 --truncate
./eval.py --model ww4bn2-conv1-halfup --net ww4bn2 --truncate
