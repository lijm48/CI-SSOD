#from mmdet.datasets import DATASETS, ConcatDataset, build_dataset
from mmdet.datasets import DATASETS, ConcatDataset
from .builder import build_dataset
import numpy as np
import torch
import os
import bisect
import collections
import copy
import math
from collections import defaultdict

from mmcv.utils import build_from_cfg, print_log
from .cate import COCO_NUM_IMAGE, COCO_NUM_IMAGE_2, OBJECT365_NUM_IMAGE

@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup), build_dataset(unsup)], **kwargs)

    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]

@DATASETS.register_module()
class SemisplitDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup_base: dict, sup_novel:dict, unsup: dict, sup:dict, **kwargs):
        super().__init__([build_dataset(sup_base), build_dataset(sup_novel), build_dataset(unsup)], **kwargs)

    @property
    def sup_base(self):
        return self.datasets[0]

    @property
    def sup_novel(self):
        return self.datasets[0]


    @property
    def unsup(self):
        return self.datasets[1]


# Modified from https://github.com/facebookresearch/detectron2/blob/41d475b75a230221e21d9cac5d69655e3415e3a4/detectron2/data/samplers/distributed_sampler.py#L57 # noqa
@DATASETS.register_module()
class ClassBalancedDatasetRFS:
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will not be oversampled. Otherwise, they will be categorized
            as the pure background class and involved into the oversampling.
            Default: True.
    """

    def __init__(self, dataset, oversample_thr, filter_empty_gt=True):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = dataset.CLASSES

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):                                                                                                                                                 
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        # category_repeat = {
        #     cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
        #     for cat_id, cat_freq in category_freq.items()
        # }
        category_repeat = {
            cat_id: max(1.0, math.pow((repeat_thr / cat_freq), 0.5))
            for cat_id, cat_freq in category_freq.items()
        }
        print("use frequency factor 0.5.")
        # category_repeat = {
        #     cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
        #     for cat_id, cat_freq in category_freq.items()
        # }
        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)



@DATASETS.register_module()
class ClassBalancedDatasetCrS:
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will not be oversampled. Otherwise, they will be categorized
            as the pure background class and involved into the oversampling.
            Default: True.
    """

    def __init__(self, dataset, filter_empty_gt=True, split='lvis', **kwargs):
        self.dataset = dataset
        # self.oversample_thr = oversample_thr
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = dataset.CLASSES
        if split == 'lvis':
            self.classes_fre = np.load("./lvis.npy")
            self.num_sup = 9946
        elif split == 'coco1':
            self.classes_fre = torch.tensor(COCO_NUM_IMAGE)
            self.num_sup = 10170       
        elif split == 'coco2':
            self.classes_fre = torch.tensor(COCO_NUM_IMAGE_2)
            self.num_sup = 10299   
        elif split == 'coco2object365':
            self.classes_fre = torch.tensor(OBJECT365_NUM_IMAGE)
            self.num_sup = 119790  

        repeat_factors = self._get_repeat_factors(dataset)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = 0.01
        lambd = self.dataset.get_lambd()
        repeat_factors = []

        for i in range(self.classes_fre.shape[0]):
            category_freq[i] = self.classes_fre[i]

        for idx in range(len(self.dataset)):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0:
                continue
            num_images += 1
            for cat_id in cat_ids:
                category_freq[cat_id] += 1

        for k, v in category_freq.items():
            category_freq[k] = v / (num_images + self.num_sup)
        
        category_repeat = {
            cat_id: max(1.0, math.sqrt(lambd / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        count = 0
        max_repeat = 1
        for idx in range(len(self.dataset)):
            category_scores = defaultdict(int)
            cat_ids = self.dataset.get_cat_ids(idx)
            scores = self.dataset.get_scores(idx)
            repeat_factor = 1
            for i in range(len(cat_ids)):
                cat_id = cat_ids[i]
                repeat_factor = max( category_repeat[cat_id] * scores[i], repeat_factor)
            if repeat_factor > 1:
                max_repeat = max( max_repeat, repeat_factor)
                
                count += 1
            repeat_factors.append(repeat_factor)
        print("cate num: ",len(category_repeat), " repeat count ",count, "max repeat", max_repeat, "pseudo nums", num_images)


        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)



