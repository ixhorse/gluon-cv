#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

FLAG=$1

# Set up the working directories.
VOC_ROOT="${HOME}/data/VOCdevkit"
COCO_ROOT="${HOME}/data/COCO"
TT100K_ROOT="${HOME}/data/TT100K/TT100K_voc"

echo $FLAG
if [ 1 == $FLAG ] 
then
    echo "====train===="
    python train.py \
        --model="deeplab" \
        --backbone="resnet101" \
        --dataset="tt100k" \
        --dataset_root="${TT100K_ROOT}" \
        --batch-size=4 \
        --workers=8 \
        --epochs=50 \
        --resume="" \
        --start-epoch=0 \
        --lr=1e-3 \
        --momentum=0.9 \
        --weight-decay=1e-4 \
        --aux-weight=0.5 \
        --aux \
        --syncbn \
        --no-val \
        --checkname="resnet101"
elif [ 2 == $FLAG ]
then
    echo "====test===="
    python test.py \
        --model="deeplab" \
        --backbone="resnet50" \
        --dataset="tt100k" \
        --dataset_root="${TT100K_ROOT}" \
        --resume="weights/tt100k_deeplab_resnet50_0049.params" \
        --syncbn \
        --no-val \
        --checkname="resnet101"
fi

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.