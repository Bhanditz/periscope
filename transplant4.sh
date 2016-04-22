#!/bin/sh

echo "makemerged"
./makemerged.py \
        --model1 ww4bn2 \
        --model2 ww4bn2-2 \
        --save ww4bn2-merged-0-2

echo "makemerged"
./makemerged.py \
        --model1 ww4bn2-3 \
        --model2 ww4bn2-4 \
        --save ww4bn2-merged-3-4

echo "makemerged"
./makemerged.py \
        --model1 ww4bn2-merged-0-2 \
        --model2 ww4bn2-merged-3-4 \
        --save ww4bn2-merged-0-2-3-4 \

echo "makeperm"
./makeperm.py \
        --model ww4bn2-merged-0-2-3-4 \
        --perm conv1 0 16 64 80 128 144 192 208 \
        --perm conv2 0 16 64 80 128 144 192 208 \
        --perm conv3 0 32 128 160 256 288 384 416 \
        --perm conv4 0 32 128 160 256 288 384 416 \
        --perm conv5 0 64 256 320 512 544 768 832 \
        --perm conv6 0 64 256 320 512 544 768 832 \
        --perm conv7 0 128 512 640 1024 1088 1536 1664 \
        --perm conv8 0 128 512 640 1024 1088 1536 1664 \
        --perm fc9 0 256 1024 1280 2048 2176 3072 3328 \
        --save ww4bn2-merged4-permuted

echo "activation"
./activation.py \
        --model ww4bn2-merged4-permuted \

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
        --model ww4bn2-merged4-permuted \
        --reduce conv1 64 \
        --reduce conv2 64 \
        --reduce conv3 128 \
        --reduce conv4 128 \
        --reduce conv5 256 \
        --reduce conv6 256 \
        --reduce conv7 512 \
        --reduce conv8 512 \
        --reduce fc9 1024 \
        --save ww4bn2-transplant4

echo "vote"
./vote.py --model ww4bn2-transplant4 --truncate
