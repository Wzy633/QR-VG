import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss)

from einops import rearrange

class SetCriterion(nn.Module):
    """ This class computes the loss for ReferFormer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_alpha = focal_alpha
        self.mask_out_stride = 4

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] 
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')

        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices): 
            valid_ind = valid.nonzero().flatten() 
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) 
        if self.num_classes == 1:
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            pass
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']  
        bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)  

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]  
        src_boxes = src_boxes.flatten(0, 1)

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def loss_iou_prediction(self, outputs, targets, indices, num_boxes):
        """IoU预测损失：组合加权MSE + Focal + Smooth L1 + L1"""
        assert 'pred_iou' in outputs
        pred_iou = outputs['pred_iou']
        bs, nf, nq = pred_iou.shape[:3]
        pred_iou = pred_iou.transpose(1, 2)

        idx = self._get_src_permutation_idx(indices)
        pred_iou = pred_iou[idx]
        pred_iou = pred_iou.flatten(0, 1)

        src_boxes = outputs['pred_boxes'].transpose(1, 2)[idx].flatten(0, 1)
        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)
        
        if torch.isnan(src_boxes).any() or torch.isinf(src_boxes).any():
            print(f"Warning: NaN/Inf detected in src_boxes for IoU loss, skipping this batch")
            return {'loss_iou': torch.tensor(0.0, device=pred_iou.device, requires_grad=True)}
        if torch.isnan(target_boxes).any() or torch.isinf(target_boxes).any():
            print(f"Warning: NaN/Inf detected in target_boxes for IoU loss, skipping this batch")
            return {'loss_iou': torch.tensor(0.0, device=pred_iou.device, requires_grad=True)}
        
        target_w = target_boxes[:, 2]
        target_h = target_boxes[:, 3]
        valid_target_mask = (target_w > 1e-6) & (target_h > 1e-6)
        if not valid_target_mask.all():
            num_invalid = (~valid_target_mask).sum().item()
            if num_invalid > 0:
                print(f"Warning: {num_invalid} invalid target_boxes detected (w or h <= 0), filtering them")
                valid_indices = torch.where(valid_target_mask)[0]
                if len(valid_indices) == 0:
                    return {'loss_iou': torch.tensor(0.0, device=pred_iou.device, requires_grad=True)}
                pred_iou = pred_iou[valid_indices]
                src_boxes = src_boxes[valid_indices]
                target_boxes = target_boxes[valid_indices]
        
        true_iou, _ = box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)
        )
        true_iou = torch.diag(true_iou).unsqueeze(1)
        
        if torch.isnan(true_iou).any() or torch.isinf(true_iou).any():
            print(f"Warning: NaN/Inf detected in true_iou, replacing with zeros")
            true_iou = torch.where(torch.isnan(true_iou) | torch.isinf(true_iou),
                                  torch.zeros_like(true_iou), true_iou)
            true_iou = true_iou.clamp(min=0.0, max=1.0)

        pred_iou = torch.clamp(pred_iou, 0.0, 1.0)
        
        if torch.isnan(pred_iou).any() or torch.isinf(pred_iou).any():
            print(f"Warning: NaN/Inf detected in pred_iou, replacing with zeros")
            pred_iou = torch.where(torch.isnan(pred_iou) | torch.isinf(pred_iou),
                                  torch.zeros_like(pred_iou), pred_iou)
            pred_iou = pred_iou.clamp(min=0.0, max=1.0)

        loss_mse = F.mse_loss(pred_iou, true_iou, reduction='none')
        iou_weights = torch.ones_like(true_iou)
        iou_weights = torch.where(true_iou < 0.5, torch.tensor(1.5, device=true_iou.device), iou_weights)
        iou_weights = torch.where((true_iou >= 0.5) & (true_iou < 0.8), torch.tensor(2.0, device=true_iou.device), iou_weights)
        iou_weights = torch.where(true_iou >= 0.8, torch.tensor(1.0, device=true_iou.device), iou_weights)
        loss_weighted = (iou_weights * loss_mse).mean()

        error = torch.abs(pred_iou - true_iou)
        focal_weight = (error + 0.1) ** 2.0
        focal_range = torch.ones_like(true_iou)
        focal_range = torch.where(true_iou < 0.5, torch.tensor(1.5, device=true_iou.device), focal_range)
        focal_range = torch.where((true_iou >= 0.5) & (true_iou < 0.8), torch.tensor(2.0, device=true_iou.device), focal_range)
        focal_range = torch.where(true_iou >= 0.8, torch.tensor(1.0, device=true_iou.device), focal_range)
        focal_weight = focal_weight * focal_range
        loss_focal = (focal_weight * loss_mse).mean()

        loss_smooth_l1 = F.smooth_l1_loss(pred_iou, true_iou)

        loss_l1 = F.l1_loss(pred_iou, true_iou)

        loss_iou = 0.45 * loss_weighted + 0.35 * loss_focal + 0.15 * loss_smooth_l1 + 0.05 * loss_l1
        return {'loss_iou': loss_iou}

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)

        src_masks = outputs["pred_masks"] 
        src_masks = src_masks.transpose(1, 2) 

        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                              size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks) 

        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        src_masks = src_masks[src_idx] 
        src_masks = src_masks.flatten(1)

        target_masks = target_masks.flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'iou': self.loss_iou_prediction
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1)
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


