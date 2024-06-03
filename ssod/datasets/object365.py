# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose
import gc
from .cate import  COCO_OBJECT365_CATEGORIES, COCO_INDEX, OBJECT365_CATEGORIES 
from mmdet.datasets import CocoDataset
import json

@DATASETS.register_module()
class SemiObject365Dataset(CocoDataset):

    CLASSES = OBJECT365_CATEGORIES 

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 is_labeled = True,
                 get_pseudo_labeled = False,
                 pseudo_ann_file = None,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.is_labeled = is_labeled
        self.get_pseudo_labeled = get_pseudo_labeled
        self.pseudo_ann_file = pseudo_ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self._base_classes = [0, 46, 5, 58, 114, 55, 116, 65, 21, 40, 176, 127, 249, \
            24, 56, 139, 92, 78, 99, 96, 144, 295, 178, 180, 38, 39, 13, 43, 120, 219, 119, 173, \
            154, 137, 113, 145, 146, 204, 8, 35, 10, 88, 84, 93, 26, 112, 82, 265, 104, 141, 152, 234, \
            143, 150, 97, 2, 50, 25, 75, 98, 153, 37, 73, 115, 132, 106, 61, 163, 134, 277, 81, 133, 18, \
            94, 30, 169, 328, 226]
        for i in range(len(self._base_classes)):
            self._base_classes[i] = self._base_classes[i] + 1
        self._novel_classes = []

        for i in range(365):
            if i+1 not in self._base_classes:
                self._novel_classes.append(i+1)

        self.novel_ids = self._novel_classes
        self.names = OBJECT365_CATEGORIES
        self.file_client = mmcv.FileClient(**file_client_args)
        self.annos = []
        self.classes_num = [0 for i in range(365)]
        self.count = 0

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path, pseudo_ann_file)
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file, pseudo_ann_file)

        if self.proposal_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.ann_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None



        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            self.annos = [self.annos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

            del self.coco
            gc.collect()
        # processing pipeline
        self.pipeline = Compose(pipeline)
        

    def load_annotations(self, ann_file, pseudo_ann_file = None):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        #self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        if self.get_pseudo_labeled:
            self.check_and_create_files(pseudo_ann_file)
            with open(pseudo_ann_file) as f:
                pseudo_json = json.load(f)
            if 'lambd' in pseudo_json.keys():
                self.lambd = pseudo_json['lambd']
            else:
                self.lambd = 0
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            if not self.test_mode:
                img_id = info['id']

                assert img_id == i, "img_id do not equal to i"
                if not self.is_labeled:
                    # due with some bad cases in unlabeled images which cause errors
                    if info["file_name"][10:] == 'patch6/objects365_v1_00320534.jpg' or info["file_name"][10:] == 'patch6/objects365_v1_00320532.jpg' \
                    or info["file_name"][10:] == 'patch16/objects365_v2_00908726.jpg':
                        continue
                    if info["file_name"][10:] == "patch44/objects365_v2_02024631.jpg" or info["file_name"][10:] ==  'patch21/objects365_v2_01118660.jpg' \
                    or info["file_name"][10:] == 'patch10/objects365_v1_00509398.jpg':
                        width = info['width']
                        info['width'] =  info['height']
            info['type'] = 'sup'
                    
            info['filename'] = info['file_name'][10:]
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            if not self.test_mode and not self.is_labeled:
                if self.get_pseudo_labeled:
                    anno = self._parse_pseudo_ann_info(info, pseudo_json)
                else: 
                    anno = {}
            else:
                anno = self._parse_ann_info(info, ann_info)
            self.annos.append(anno)    


            

        print("Length of dataset:", len(data_infos))
        
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"


        return data_infos


    def check_and_create_files(self, file_path):
        directory = osp.dirname(file_path)
        if not osp.exists(directory):
            try:
                os.makedirs(directory) 
                print(f"directory {directory} created!")
            except Exception as e:
                print(f"directory creation failed:{e}")
        if not osp.exists(file_path):
            try:
                with open(file_path, 'w') as f:
                    json.dump({}, f)  
                print(f"file {file_path} created!")
            except Exception as e:
                print(f"file creation failed:{e}")

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        return self.annos[idx]


    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        return self.annos[idx]['category_ids']

    def get_lambd(self):
        return self.lambd

    def get_scores(self, idx):
        return self.annos[idx]['scores']

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.data_infos[i]['id']
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_pseudo_ann_info(self, img_info, pse_annos):
        file_name = img_info['filename']
        file_name = osp.join(self.img_prefix, file_name)
        
        if file_name not in pse_annos:
            self.count += 1
            return {'scores':{}, 'category_ids': {}, 'bboxes': np.zeros((0, 4)),'labels': np.zeros((0,))}
        info = pse_annos[file_name]
        anno = info['annotations']
        category_ids = []
        scores = []
        for j in range(len(anno)):
            category_ids.append(anno[j]['category_id'])
            if 'score' in anno[j]:
                scores.append(anno[j]['score'])
            else:
                scores.append(1)

        ann = dict(scores = scores, category_ids = category_ids,  bboxes=np.zeros((0, 4)),labels=np.zeros((0,)))
        return ann

    def _parse_ann_info(self, img_info, ann_info, is_novel = False):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        category_ids = [] 
        flag = 0
        vis = [0 for i in range(365)]
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            if vis[self.cat2label[ann['category_id']]] == 0:
                self.classes_num[self.cat2label[ann['category_id']]]+=1
                vis[self.cat2label[ann['category_id']]] = 1
            category_ids.append(self.cat2label[ann['category_id']])
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            category_ids = category_ids,
            bboxes_ignore=gt_bboxes_ignore)
        return ann




    def evaluate_det_segm_bn(self,
                            results,
                            result_files,
                            coco_gt,
                            metrics,
                            logger=None,
                            classwise=True,
                            proposal_nums=(100, 300, 1000),
                            iou_thrs=None,
                            metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                if isinstance(results[0], tuple):
                    raise KeyError('proposal_fast is not supported for '
                                   'instance segmentation result.')
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(coco_gt, coco_det, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                for split, classes in [
                        ("all",   self.cat_ids),
                        ("base", self._base_classes),
                        ("novel", self._novel_classes)]:
                    # if classes != None:
                    cocoEval = COCOeval(coco_gt, coco_det, iou_type)
                    #cocoEval.params.catIds = self.cat_ids
                    cocoEval.params.imgIds = self.img_ids
                    cocoEval.params.maxDets = list(proposal_nums)
                    cocoEval.params.iouThrs = iou_thrs
                    cocoEval.params.catIds = classes
                    
                    cocoEval.evaluate()
                    cocoEval.accumulate()

                    # Save coco summarize print information to logger
                    redirect_string = io.StringIO()
                    with contextlib.redirect_stdout(redirect_string):
                        cocoEval.summarize()
                    print_log('\n' + redirect_string.getvalue(), logger=logger)
                    
                    if classwise:  # Compute per-category AP
                        # Compute per-category AP
                        # from https://github.com/facebookresearch/detectron2/
                        precisions = cocoEval.eval['precision']
                        recalls = cocoEval.eval['recall']
                        scores = cocoEval.eval['scores']
                        # precision: (iou, recall, cls, area range, max dets)
                        # assert len(self.cat_ids) == precisions.shape[2]
                        results_per_category = []
 
                        for idx, catId in enumerate(classes):
                            # area range index 0: all area ranges
                            # max dets index -1: typically 100 per image
                            name = OBJECT365_CATEGORIES[catId-1]
                            precision = precisions[:, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            results_per_category.append(
                                (f'{name}', f'{float(ap):0.3f}'))
    
                        num_columns = min(6, len(results_per_category) * 2)
                        results_flatten = list(
                            itertools.chain(*results_per_category))
                        headers = ['category', 'AP'] * (num_columns // 2)
                        results_2d = itertools.zip_longest(*[
                            results_flatten[i::num_columns]
                            for i in range(num_columns)
                        ])
                        table_data = [headers]
                        table_data += [result for result in results_2d]
                        table = AsciiTable(table_data)
                        print_log('\n' + table.table, logger=logger)
                    if split != "all":
                        continue
                    if metric_items is None:
                        metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                        ]

                    for metric_item in metric_items:
                        key = f'{metric}_{metric_item}'
                        val = float(
                            f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                        )
                        eval_results[key] = val
                    
                    ap = cocoEval.stats[:6]
                    eval_results[f'{metric}_mAP_copypaste'] = (
                        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                        f'{ap[4]:.3f} {ap[5]:.3f}')

        return eval_results


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        # self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)
        self.cat_ids = coco_gt.get_cat_ids()
        # print(len(self.cat_ids))
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm_bn(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results