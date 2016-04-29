#!/bin/sh

# echo "makemerged"
# ./makemerged.py \
#         --model1 ww4bn2 \
#         --model2 ww4bn2-2 \
#         --save ww4bn2-merged-0-2

# echo "activation"
# ./activation.py \
#         --model ww4bn2-merged-0-2order \

# echo "makehalf"
# ./makehalf.py \
#         --model ww4bn2-merged-0-2order \
#         --reduce conv1 128 \
#         --reduce conv2 128 \
#         --reduce conv3 256 \
#         --reduce conv4 256 \
#         --reduce conv5 512 \
#         --reduce conv6 512 \
#         --reduce conv7 1024 \
#         --reduce conv8 1024 \
#         --reduce fc9 1536 \
#         --save ww4bn2-augment

# Half of some layers
echo "half conv3+below, random"
./makehalf.py \
        --model ww4bn2-merged-0-2 \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 256 \
        --reduce conv5 512 \
        --reduce conv6 512 \
        --reduce conv7 1024 \
        --reduce conv8 1024 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv3+below, ordered"
./makehalf.py \
        --model ww4bn2-merged-0-2order \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 256 \
        --reduce conv5 512 \
        --reduce conv6 512 \
        --reduce conv7 1024 \
        --reduce conv8 1024 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv4+below, random"
./makehalf.py \
        --model ww4bn2-merged-0-2 \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 192 \
        --reduce conv5 512 \
        --reduce conv6 512 \
        --reduce conv7 1024 \
        --reduce conv8 1024 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv4+below, ordered"
./makehalf.py \
        --model ww4bn2-merged-0-2order \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 192 \
        --reduce conv5 512 \
        --reduce conv6 512 \
        --reduce conv7 1024 \
        --reduce conv8 1024 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv5+below, random"
./makehalf.py \
        --model ww4bn2-merged-0-2 \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 192 \
        --reduce conv5 384 \
        --reduce conv6 512 \
        --reduce conv7 1024 \
        --reduce conv8 1024 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv5+below, ordered"
./makehalf.py \
        --model ww4bn2-merged-0-2order \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 192 \
        --reduce conv5 384 \
        --reduce conv6 512 \
        --reduce conv7 1024 \
        --reduce conv8 1024 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv6+below, random"
./makehalf.py \
        --model ww4bn2-merged-0-2 \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 192 \
        --reduce conv5 394 \
        --reduce conv6 384 \
        --reduce conv7 1024 \
        --reduce conv8 1024 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv6+below, ordered"
./makehalf.py \
        --model ww4bn2-merged-0-2order \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 192 \
        --reduce conv5 384 \
        --reduce conv6 384 \
        --reduce conv7 1024 \
        --reduce conv8 1024 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv7+below, random"
./makehalf.py \
        --model ww4bn2-merged-0-2 \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 192 \
        --reduce conv5 394 \
        --reduce conv6 384 \
        --reduce conv7 768 \
        --reduce conv8 1024 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv7+below, ordered"
./makehalf.py \
        --model ww4bn2-merged-0-2order \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 192 \
        --reduce conv5 384 \
        --reduce conv6 384 \
        --reduce conv7 768 \
        --reduce conv8 1024 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv8+below, random"
./makehalf.py \
        --model ww4bn2-merged-0-2 \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 192 \
        --reduce conv5 394 \
        --reduce conv6 384 \
        --reduce conv7 768 \
        --reduce conv8 768 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate

echo "half conv8+below, ordered"
./makehalf.py \
        --model ww4bn2-merged-0-2order \
        --reduce conv1 96 \
        --reduce conv2 96 \
        --reduce conv3 192 \
        --reduce conv4 192 \
        --reduce conv5 384 \
        --reduce conv6 384 \
        --reduce conv7 768 \
        --reduce conv8 768 \
        --reduce fc9 2048 \
        --save ww4bn2-augment

# Half of every layer
# ./makehalf.py \
#         --model ww4bn2-merged-0-2 \
#         --reduce conv1 96 \
#         --reduce conv2 96 \
#         --reduce conv3 192 \
#         --reduce conv4 192 \
#         --reduce conv5 384 \
#         --reduce conv6 384 \
#         --reduce conv7 768 \
#         --reduce conv8 768 \
#         --reduce fc9 1536 \
#         --save ww4bn2-augment

# No augmentation
# ./makehalf.py \
#         --model ww4bn2-merged-0-2 \
#         --reduce conv1 64 \
#         --reduce conv2 64 \
#         --reduce conv3 256 \
#         --reduce conv4 256 \
#         --reduce conv5 512 \
#         --reduce conv6 512 \
#         --reduce conv7 1024 \
#         --reduce conv8 768 \
#         --reduce fc9 1536 \
#         --save ww4bn2-augment

echo "vote"
./vote.py --model ww4bn2-augment --truncate
