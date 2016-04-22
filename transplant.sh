#!/bin/sh

echo "makemerged"
./makemerged.py \
        --model1 ww4bn2 \
        --model2 ww4bn2-2 \
        --save ww4bn2-merged-0-2

echo "makeperm"
./makeperm.py \
        --model ww4bn2-merged-0-2 \
        --perm conv1 0 32 64 96 \
        --perm conv2 0 32 64 96 \
        --perm conv3 0 64 128 192 \
        --perm conv4 0 64 128 192 \
        --perm conv5 0 128 256 384 \
        --perm conv6 0 128 256 384 \
        --perm conv7 0 256 512 768 \
        --perm conv8 0 256 512 768 \
        --perm fc9 0 512 1024 1536 \
        --save ww4bn2-merged-permuted

# echo "activation"
# ./activation.py \
#         --model ww4bn2-merged-permuted \

echo "makehalf"
# ./makehalf.py \
#         --model ww4bn2-merged-permuted \
#         --reduce conv1 128 \
#         --reduce conv2 128 \
#         --reduce conv3 256 \
#         --reduce conv4 256 \
#         --reduce conv5 512 \
#         --reduce conv6 512 \
#         --reduce conv7 1024 \
#         --reduce conv8 1024 \
#         --reduce fc9 2048 \
#         --save ww4bn2-transplant

./makehalf.py \
        --model ww4bn2-merged-permuted \
        --reduce conv1 64 \
        --reduce conv2 64 \
        --reduce conv3 128 \
        --reduce conv4 128 \
        --reduce conv5 256 \
        --reduce conv6 256 \
        --reduce conv7 512 \
        --reduce conv8 512 \
        --reduce fc9 1024 \
        --save ww4bn2-transplant

echo "vote"
./vote.py --model ww4bn2-transplant --truncate
