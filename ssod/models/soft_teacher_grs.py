import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid, filter_invalid_dynamic
from mmcv.runner import get_dist_info
import numpy as np
import random
import json
import time
import torch.nn.functional as F
import copy
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)

# from ssod.core.post_processing import  multiclass_nms



@DETECTORS.register_module()
class SoftTeacherGrS(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacherGrS, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
            self.pseudo_pools = {} 
            self.tmp_pseudo_pools = {}  
            self.pseudo_dir = self.train_cfg.pseudo_dir

            self.alpha =  self.train_cfg.alpha
            self.max_iter = self.train_cfg.max_iter
            self.warm_up_iter = self.train_cfg.warm_up_iter
            self.save_iter = 4000
            self.ema_factor = 0.999        

        

        self.num_classes = self.student.roi_head.bbox_head.num_classes
        self.iter = 0
        self.thres = []



    

        self.class_weights = 1
        self.cls_scores = torch.zeros(self.num_classes)
        self.cls_nums = torch.zeros(self.num_classes)



    def train_step(self, data, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data, **kwargs)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

        
    
                       
    def sync_pse(self, iter = 0):
        """
        It seems torch.distributed do not support sync for dict, so here we save the pseudo labels on disks and merge them to a single file.
        """
        rank, world_size = get_dist_info()

        torch.distributed.barrier()
        if rank == 0:
            
            print("world_size:", world_size)
            st = time.time()
            new_dict = {}
            for i in range(world_size):
                with open(self.pseudo_dir+"pseudo_"+str(i)+".json") as f:
                    pseudo_json = json.load(f)
                new_dict.update(pseudo_json)
            with open(self.pseudo_dir+"/pseudo.json") as f:
                cur_dict = json.load(f)
            cur_dict.update(new_dict)
            cur_dict['lambd'] = min(self.alpha * self.iter / self.max_iter, self.alpha)
            with open(self.pseudo_dir+"/pseudo.json","w") as f:
                json.dump(cur_dict, f)         

            with open(self.pseudo_dir+"/pseudo_"+str(self.iter)+".json","w") as f:
                json.dump(cur_dict, f)  
            torch.distributed.barrier() 
            et = time.time()
            print("sync cost: "+ str(et-st))
        else:
            torch.distributed.barrier()

        # print(self.iter, self.thres)
        # print(self.class_weights)
            
        self.tmp_pseudo_pools = {}     


    def save_pse(self, iter = 0):
        print("saving pseudo json")
        rank, world_size = get_dist_info()
        new_dict = self.tmp_pseudo_pools

        with open(self.pseudo_dir+"/pseudo_"+str(rank)+".json","w") as f:
            json.dump(new_dict, f)
 
        self.sync_pse()

        
         


    def forward_train(self, img, img_metas, iter_num = None, **kwargs):
        super().forward_train(img, img_metas, **kwargs)

        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}

        self.iter = iter_num
        if iter_num % self.save_iter == 1:
            self.save_pse(iter_num)

        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            #sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = self.forward_sup_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def forward_sup_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.student.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.student.with_rpn:
            proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                              self.student.test_cfg.rpn)
            
            rpn_losses, proposal_list = self.student.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self.student.roi_head._bbox_forward(x, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.roi_head.train_cfg
        )
        labels = bbox_targets[0]
        scores = F.softmax(bbox_results["cls_score"], dim = 1)

        roi_losses = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            iter = self.iter,
            loss_type = 0,
            loss_weight = 1,
        )

        losses.update(roi_losses)

        return losses


    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        loss.update(
            self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        )
        loss.update(
            self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
            )
        )
        return loss

    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:, :4],
                bbox_tag="rpn_pseudo_label",
                scores=pseudo_bboxes[0][:, 4],
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        if self.iter < self.warm_up_iter:
            gt_bboxes, gt_labels, _ = multi_apply(
                filter_invalid,
                [bbox[:, :4] for bbox in pseudo_bboxes],
                pseudo_labels,
                [bbox[:, 4] for bbox in pseudo_bboxes],
                thr=self.train_cfg.cls_pseudo_threshold,
            )
        else:
            gt_bboxes, gt_labels, _ = multi_apply(
                filter_invalid_dynamic,
                [bbox[:, :4] for bbox in pseudo_bboxes],
                pseudo_labels,
                [bbox[:, 4] for bbox in pseudo_bboxes],
                thr=self.thres,
            )

        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()

        # loss with GbR
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
            iter = self.iter,
            loss_type = 1,
            loss_weight = self.unsup_weight,
        )
        if "loss_cls" in loss.keys():
            loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        else:
            print("no loss cls in unsup!")

        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss

    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox}

    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes


    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox_inverse(self, bboxes, trans_mat, max_shape):
        trans_mat = [at.inverse() for at in trans_mat]
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.teacher.roi_head.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self.teacher.roi_head._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.teacher.roi_head.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label, indexes = self.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels, cls_score[indexes]


    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                Fisrt tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.teacher.roi_head.bbox_head.custom_cls_channels:
            scores = self.teacher.roi_head.bbox_head.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.teacher.roi_head.bbox_head.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:

            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels, indexes = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, return_inds = True)

            return det_bboxes, det_labels, indexes

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        # img_metas[0]['transform_matrix']
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        # proposal_list, proposal_label_list, proposal_scores = self.simple_test_bboxes(
        #     feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        # )
        proposal_list, proposal_label_list= self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )
        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        # self.tmp_fre = self.tmp_fre.to(feat[0].device)

        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        self.cls_scores  = self.cls_scores.to(feat[0].device)
        self.cls_nums = self.cls_nums.to(feat[0].device)
        class_weights = self.student.roi_head.bbox_head.get_class_weights()
        for i in range(len(proposal_list)):
            for j in range(proposal_list[i].shape[0]):
                tmp_score = proposal_list[i][j, -1]
                pred_idx = proposal_label_list[i][j].item()
                if self.cls_nums[pred_idx] < 1000:
                    self.cls_scores[pred_idx] = (self.cls_scores[pred_idx] * 1000 + tmp_score) / 1000
                    self.cls_nums[pred_idx] += 1
                else:
                    self.cls_scores[pred_idx] = self.ema_factor *  self.cls_scores[pred_idx] + (1 - self.ema_factor) * tmp_score 

        all_score = self.cls_scores.mean()
        self.thres = self.train_cfg.cls_pseudo_threshold * (self.cls_scores / all_score)
        
        if class_weights != None and self.train_cfg.GbT:
            self.thres = torch.min(self.thres, self.train_cfg.cls_pseudo_threshold / class_weights[0:self.num_classes].to(self.thres.device))
            assert class_weights.shape[0] == self.num_classes+1, "The dim of class_weights is wrong!"
        self.class_weights = class_weights
        
        self.thres = self.thres.clamp(self.train_cfg.pseudo_label_initial_score_thr, self.train_cfg.cls_pseudo_threshold)

        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")

        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )




        det_bboxes = proposal_list
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]

        aligned_proposal = [res[:, :4] for res in proposal_list]

        aligned_proposal = self._transform_bbox_inverse(
            aligned_proposal,
            teacher_info["transform_matrix"],
            [meta["img_shape"] for meta in img_metas],
        )

        #dump the pseudo labels in pools
        if self.iter > self.warm_up_iter:
            for i in range(len(proposal_list)):
                count = 0
                annotations = []

                for j in range(proposal_list[i].shape[0]):
                    bbox = aligned_proposal[i][j, :4].cpu().numpy().tolist()
                    tmp_score = proposal_list[i][j, -1]
                    min_size =self.train_cfg.min_pseduo_box_size
                    anno = {}
                    pred_idx = proposal_label_list[i][j].item()
                    anno['category_id'] = pred_idx
                    anno['bbox'] = bbox
                    
                    if bbox[2] - bbox[0] <= min_size or bbox[3] - bbox[1] <= min_size:
                        continue


                    if tmp_score < self.thres[pred_idx]:
                        continue
                    anno['score'] = tmp_score.item() - self.thres[pred_idx].item()
                    #((tmp_score - self.thres[pred_idx]) / (1 - self.thres[pred_idx])).item()
                    annotations.append(anno)
                    
                file_name = img_metas[i]['filename']
                images_info = {}
                images_info['file_name'] = file_name
                images_info['width'] = img_metas[i]['ori_shape'][1]
                images_info['height'] = img_metas[i]['ori_shape'][0]
                images_info['annotations'] = annotations

                self.tmp_pseudo_pools[images_info['file_name']] = images_info




        teacher_info["img_metas"] = img_metas
        return teacher_info




    def compute_uncertainty_with_aug(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc




    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )