"""
This module provides data loaders and transformers for popular vision datasets.
"""
from . import transforms
from . import batchify
from .imagenet.classification import ImageNet, ImageNet1kAttr
from .dataloader import DetectionDataLoader, RandomTransformDataLoader
from .pascal_voc.detection import VOCDetection
from .mscoco.detection import COCODetection
from .mscoco.instance import COCOInstance
from .mscoco.segmentation import COCOSegmentation
from .mscoco.keypoints import COCOKeyPoints
from .cityscapes import CitySegmentation
from .pascal_voc.segmentation import VOCSegmentation
from .pascal_aug.segmentation import VOCAugSegmentation
from .ade20k.segmentation import ADE20KSegmentation
from .segbase import ms_batchify_fn
from .recordio.detection import RecordFileDetection
from .lst.detection import LstDetection
from .mixup.detection import MixupDetection
from .tt100k.detection import TT100KDetection
from .tt100k.segmentation import TT100KSegmentation
from .tt100k.region import TT100KRegion

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'coco' : COCOSegmentation,
    'citys' : CitySegmentation,
    'tt100k' : TT100KSegmentation,
    'tt100k_region' : TT100KRegion,
}

def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
