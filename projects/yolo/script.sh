#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

FLAG=$1

# Set up the working directories.
VOC_ROOT="${HOME}/data/VOCdevkit"
COCO_ROOT="${HOME}/data/COCO"
TT100K_ROOT="${HOME}/data/TT100K/TT100K_chip_voc"

echo $FLAG
if [ 1 == $FLAG ] 
then
    echo "====train===="
    python train_yolo3.py \
        --network="darknet53" \
        --batch-size=16 \
        --dataset="tt100k" \
        --dataset_root="${TT100K_ROOT}" \
        --num-workers=8\
        --gpus="0,1" \
        --epochs=300 \
        --resume="" \
        --start-epoch=0 \
        --lr=1e-3 \
        --lr-decay-epoch="250" \
        --momentum=0.9 \
        --wd=5e-4 \
        --val=0 \
        --warmup-epochs=4 \
        --syncbn
elif [ 2 == $FLAG ]
then
    echo "====test===="
    python test_yolo.py \
        --network="darknet53" \
        --dataset="tt100k" \
        --dataset_root="${TT100K_ROOT}" \
        --pretrained="weights/yolo3_darknet53_custom_0250.params"
else
    echo "error"
fi

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
