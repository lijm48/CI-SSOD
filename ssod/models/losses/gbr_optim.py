import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmdet.utils import get_root_logger
from functools import partial
import numpy as np
from mmdet.models.builder import LOSSES
from mmcv.runner import get_dist_info
import os
from mmdet.models.losses.utils import weight_reduce_loss

from ssod.utils import check_and_create_path

@LOSSES.register_module()
class GbROptim(nn.Module):
    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=1203,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 test_with_obj=True,
                 beta_sup = 0.5,
                 warm_up_iter = 8000,
                 ramp_up_iter = 0,
                 upper_bound = 5,
                 lower_bound = 1/3,
                 gamma=1.0,
                 use_sigmoid = False,
                 unsup_loss_weight=4,
                 focal=True):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes

        # cfg for GbR
        self.warm_up_iter = warm_up_iter
        self.ramp_up_iter = ramp_up_iter
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.beta_sup = beta_sup
        self.focal = focal #focal loss or cross entropy


        self.max_iter = 0
        self.iter = 0
        self.ema_factor = 0.999
        self.save_iter = 1000
        self.gamma = gamma
        self.class_weights = torch.ones(num_classes+1)
        self.target_class_weights = torch.ones(num_classes+1)
        self.pos_grad = torch.zeros(num_classes+1)
        self.register_buffer('corr_grad', torch.zeros((self.num_classes+1, self.num_classes+1)))
        self.register_buffer('corr_grad_accu', torch.zeros((self.num_classes+1, self.num_classes+1)))
        self.register_parameter('classes_logits', nn.Parameter(torch.zeros((self.num_classes+1))))
        self.register_buffer('weights_mean', torch.ones(1))


        self.test_with_obj = test_with_obj

        logger = get_root_logger()
        logger.info(f"build GbR optim")

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                iter = -1,
                loss_type = 1,
                loss_weight = 1,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()
        # import pdb;pdb.set_trace()
        self.iter = iter
        reduction = (
            reduction_override if reduction_override else self.reduction)
        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

             
        self.label = label
        self.target = expand_label(cls_score, label)

        self.weight = weight.float().detach()
        class_weights = None

        # loss_type: 0:supervised 1:unsupervised
        beta = self.beta_sup
        if loss_type == 1:
            beta = 1
        with torch.no_grad():
            tmp_weight = self.get_weight(cls_score, beta) 
            if weight is not None:
                tmp_weight = tmp_weight.detach() * weight

        CE = F.cross_entropy(cls_score, label, reduction='none')
        #focal loss or cross entropy
        if self.focal:
            p = torch.exp(-CE) 
            loss = (1 - p) ** self.gamma * CE 
        else:
            loss = CE

        loss = weight_reduce_loss(
            loss, weight=tmp_weight, reduction=reduction, avg_factor=avg_factor)

        if iter >= 0 :
            # collect the positive and negative gradients
            loss_grad = self.collect_grad_optim(cls_score.detach(), self.target.detach(), tmp_weight.detach(), loss_weight)
            if loss_grad == None:
                loss_grad = 0 * loss
        else:
            loss_grad = 0 * loss

        return self.loss_weight * loss, loss_grad, self.class_weights

        


    def get_activation(self, cls_score):
        cls_score = F.softmax(cls_score, dim=-1)
        return cls_score


    def collect_grad(self, cls_score, target, weight, loss_weight = 1):
        with torch.no_grad():
            prob = F.softmax(cls_score, dim = -1)
            prob = (prob.clone() + 0.00001)
            # gradients of probability of focal loss or cross entropy
            if self.focal:
                focal_grad = self.weight.unsqueeze(1) * (((1 - prob) / prob) - torch.log(prob))
            else:
                focal_grad = self.weight.unsqueeze(1) *  (1 / (prob)) 


            # positive gradients are multipled by (1-prob) * prob, which is gradients of logits with softmax
            pos_grad = ((1-prob) * prob * focal_grad)
            pos_grad = (target * pos_grad).mean(dim=0)

            # negative gradients are multipled by prob_a * prob_b, 
            correlated_grad = prob.unsqueeze(1).repeat((1, self.n_c, 1)) 
            correlated_grad = correlated_grad * prob.unsqueeze(2)
            correlated_grad = correlated_grad * (focal_grad.unsqueeze(2) * target.unsqueeze(2))
            correlated_grad = correlated_grad.mean(dim=0)


            for i in range(self.num_classes+1):
                correlated_grad[i][i] = -pos_grad[i]

            #multiple 100 to avoid error.
            self.corr_grad = correlated_grad * 100 * loss_weight
            dist.all_reduce(self.corr_grad)
        
        
        if self.iter < self.warm_up_iter :
            self.corr_grad_accu = (self.corr_grad_accu * self.iter + self.corr_grad ) / (self.iter + 1)
        else:
            self.corr_grad_accu = self.ema_factor *  self.corr_grad_accu + (1- self.ema_factor) * self.corr_grad
            self.class_weights = F.softmax(self.classes_logits, dim=0).detach()  * (self.num_classes + 1)

        loss_grad = None
        if self.iter > self.warm_up_iter :
            for i in range(self.num_classes):
                self.target_class_weights[i] = ((self.class_weights * self.corr_grad_accu[:, i]).sum() / (-self.corr_grad_accu[i, i] + 0.0001) + self.class_weights[i] ) 
            self.target_class_weights = self.target_class_weights.to(self.classes_logits.device)

            self.target_class_weights[-1] = self.num_classes + 1 - self.target_class_weights[:-1].sum()
            self.target_class_weights = self.target_class_weights.clamp(self.lower_bound, self.upper_bound)
            self.target_class_weights[-1] = self.target_class_weights[-1].clamp(self.lower_bound, 1)

            self.target_class_weights = torch.log(self.target_class_weights)
            loss_grad =  1 * ((self.target_class_weights -  self.classes_logits) * (self.target_class_weights - self.classes_logits)).mean()
        return loss_grad




    def collect_grad_optim(self, cls_score, target, weight, loss_weight = 1):
        """ 
            optimize to save GPU memory especially when classes number are very large
            the same with collect_grad
        """
        arr = torch.arange(self.num_classes + 1)
        with torch.no_grad():
            prob = F.softmax(cls_score, dim = -1)
            prob = (prob.clone() + 0.00001)
            # gradients of probability of focal loss or cross entropy
            if self.focal:
                focal_grad = self.weight.unsqueeze(1) * (((1 - prob) / prob) - torch.log(prob))
            else:
                focal_grad = self.weight.unsqueeze(1) *  (1 / (prob)) 



            # positive gradients are multipled by (1-prob) * prob, which is gradients of logits with softmax
            pos_grad = ((1-prob) * prob * focal_grad)
            pos_grad = (target * pos_grad).mean(dim=0)

            # negative gradients are multipled by prob_a * prob_b, 
            sum_correlated_grad = 0
            interval = 64
            
            for i in range(int(target.shape[0] / interval)):
                tmp_index = range(i * interval , min((i + 1) * interval, target.shape[0]))
                correlated_grad = prob[tmp_index].unsqueeze(1).repeat((1, self.n_c, 1)) 
                correlated_grad = correlated_grad * prob[tmp_index].unsqueeze(2)
                correlated_grad = correlated_grad * focal_grad[tmp_index].unsqueeze(2) * target[tmp_index].unsqueeze(2)
                sum_correlated_grad =  sum_correlated_grad + correlated_grad.sum(dim=0)

            correlated_grad = sum_correlated_grad / target.shape[0]
            correlated_grad[arr, arr] = -pos_grad
            self.corr_grad = correlated_grad * 100 * loss_weight
            dist.all_reduce(self.corr_grad)
            # import pdb;pdb.set_trace()
        
        
        if self.iter < self.warm_up_iter :
            self.corr_grad_accu = (self.corr_grad_accu * self.iter + self.corr_grad ) / (self.iter + 1)
        else:
            self.corr_grad_accu = self.ema_factor *  self.corr_grad_accu + (1 - self.ema_factor) * self.corr_grad
            self.class_weights = F.softmax(self.classes_logits, dim=0).detach()  * (self.num_classes + 1)

        self.target_class_weights = (self.class_weights.unsqueeze(1) * self.corr_grad_accu).sum(dim=0) / (-self.corr_grad_accu[arr, arr] + 0.000001) + self.class_weights

        loss_grad = None
        if self.iter > self.warm_up_iter :
            self.target_class_weights = self.target_class_weights.to(self.classes_logits.device)
            # weights for background
            self.target_class_weights[-1] = self.num_classes + 1 - self.target_class_weights[:-1].sum()
            self.target_class_weights = self.target_class_weights.clamp(self.lower_bound, self.upper_bound)
            self.target_class_weights[-1] = self.target_class_weights[-1].clamp(self.lower_bound, 1)

            self.target_class_weights = torch.log(self.target_class_weights)
            loss_grad =  1 * ((self.target_class_weights -  self.classes_logits) * (self.target_class_weights - self.classes_logits)).mean()
        return loss_grad

    def get_weight(self, cls_score, beta):

        if self.iter < 0:
            return torch.ones(cls_score.shape[0]).to(cls_score.device)

        self.class_weights = self.class_weights.to(self.corr_grad.device)
        if self.iter % self.save_iter == 0:
            check_and_create_path("./pseudo/grad/corr_grad_accu.pth")
            torch.save(self.corr_grad_accu, "./pseudo/grad/corr_grad_accu.pth")
            if self.iter >= self.warm_up_iter :
                print("class_weights:", self.class_weights)
                torch.save(self.class_weights, "./pseudo/grad/grad_weight.pth" )
                torch.save(self.class_weights, "./pseudo/grad/grad_weight_"+str(self.iter)+".pth",)
                self.class_weights = F.softmax(self.classes_logits, dim=0).detach()  * (self.num_classes + 1)
            
        # if self.iter < self.ramp_up_iter:
        #     bound = 1 + (self.iter / self.ramp_up_iter) * (self.bound - 1)
        # else:
        #     bound = self.bound
        self.class_weights = self.class_weights.clamp(self.lower_bound, self.upper_bound) 


        tmp_weight = ((self.class_weights ** beta).unsqueeze(0).repeat(cls_score.shape[0], 1) * self.target).sum(dim = 1)
        # to ensure the weight stable 
        self.weights_mean = self.ema_factor * self.weights_mean + (1 - self.ema_factor) * tmp_weight.mean()
        tmp_weight = tmp_weight / self.weights_mean
        return tmp_weight


