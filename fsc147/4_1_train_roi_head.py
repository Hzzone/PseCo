import os
import sys

project_root = '/home/zzhuang/PseCo'
sys.path.insert(0, project_root)
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
import tqdm
import albumentations as A
import torch.nn as nn
import torchvision
import torchvision.ops as vision_ops
from ops.ops import _nms, plot_results, convert_to_cuda

import argparse

parser = argparse.ArgumentParser('Default arguments for training of different methods')
parser.add_argument('--wandb', help='wandb', action='store_true')
parser.add_argument('--zeroshot', help='zeroshot', action='store_true')
parser.add_argument('--arch', help='arch: vitb, vitl, vith', type=str, default='vith')
parser.add_argument('--entity', help='wandb user name', type=str, default='zzhuang')
opts = parser.parse_args()
print(opts)

from detectron2.data.datasets import register_coco_instances

register_coco_instances("fsc_test_val", {}, f"{project_root}/data/fsc147/instances_test_val_bin.json",
                        f"{project_root}/data/fsc147/images_384_VarV2")

torch.autograd.set_grad_enabled(False)

from ops.foundation_models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, \
    build_sam, build_sam_vit_b, build_sam_vit_h

# sam = build_sam_vit_b().cuda().eval()
# all_data = torch.load(f'{project_root}/data/fsc147/sam/all_data_vitb.pth', map_location='cpu')
# all_predictions = torch.load(f'{project_root}/data/fsc147/sam/all_predictions_vitb.pth', map_location='cpu')

# load all features and proposals
if opts.arch == 'vith':
    sam = build_sam_vit_h().cuda().eval()
    all_data = torch.load(f'{project_root}/data/fsc147/sam/all_data_vith.pth', map_location='cpu')
    all_predictions = torch.load(f'{project_root}/data/fsc147/sam/all_predictions_vith.pth', map_location='cpu')
else:
    raise NotImplementedError

# load text prompts and pseudo labels
clip_text_prompts = torch.load(f'{project_root}/data/fsc147/clip_text_prompt.pth', map_location='cpu')
all_pseudo_boxes = torch.load(f'{project_root}/data/fsc147/sam/pseudo_boxes_data_vith.pth', map_location='cpu')
for fname in tqdm.tqdm(all_data):
    target = all_data[fname]
    target['image_id'] = fname
    target['predictions'] = all_predictions[fname]
    if all_data[fname]['split'] == 'train':
        target['annotations']['boxes'] = all_pseudo_boxes[fname]['pred_boxes']
        target['annotations']['ious'] = all_pseudo_boxes[fname]['pred_ious']
all_image_list = {'train': [], 'val': [], 'test': [], 'all': []}
for fname in all_data:
    all_image_list[all_data[fname]['split']].append(fname)
    all_image_list['all'].append(fname)

from models import ROIHeadMLP as ROIHead

num_masks = 5
run_name = 'MLP_small_box_w1'
if opts.zeroshot:
    run_name += '_zeroshot'
cls_loss2_weight = 1.0

from ops.loggerx import LoggerX

logger = LoggerX(save_root=f'{project_root}/data/fsc147/checkpoints/cls_head/ckpt/{run_name}',
                 print_freq=10,
                 name=run_name,
                 enable_wandb=opts.wandb,
                 config=opts,
                 entity=opts.entity,
                 project='Counting')
cls_head = ROIHead().cuda()
logger.modules = [cls_head, ]
optimizer = torch.optim.AdamW(list(cls_head.parameters()), lr=0.0001, weight_decay=0.00001)
acc_grd_step = 1
max_iter = 10000
bs = 32

from ops.grad_scaler import NativeScalerWithGradNormCount

amp = True
scaler = NativeScalerWithGradNormCount(amp=amp)


def evaluate(split, results, threshold=None):
    image_list = [fname for fname in all_data if all_data[fname]['split'] == split]
    from detectron2.evaluation import COCOEvaluator

    coco_evaluator = COCOEvaluator(dataset_name=f'fsc_test_val', tasks=['bbox', ],
                                   output_dir=f'{project_root}/data/temp', max_dets_per_image=1000)
    coco_evaluator.reset()

    from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes

    all_predictions = {}

    for fname in tqdm.tqdm(image_list):
        features = all_data[fname]['features'].cuda()
        with torch.no_grad():
            cls_head.eval()
            if opts.zeroshot:
                example_features = clip_text_prompts[all_data[fname]['class_name']].unsqueeze(0).cuda()
            else:
                example_features = all_data[fname]['example_clip_features'].cuda()

        min_scores = 0.05
        max_points = 1000
        pred_points_score = all_data[fname]['predictions']['pred_points_score']
        mask = torch.zeros(pred_points_score.size(0))
        mask[:min(pred_points_score.size(0), max_points)] = 1
        mask[pred_points_score < min_scores] = 0
        pred_boxes = all_data[fname]['predictions']['pred_boxes'][:, :num_masks][mask.bool()].cuda()
        pred_ious = all_data[fname]['predictions']['pred_ious'][:, :num_masks][mask.bool()].cuda()

        all_pred_boxes = []
        all_pred_scores = []
        for indices in torch.arange(len(pred_boxes)).split(128):
            with torch.no_grad():
                cls_outs_ = cls_head(features, [pred_boxes[indices], ], [example_features, ] * len(indices))
                pred_logits = cls_outs_.sigmoid().view(-1, len(example_features), num_masks).mean(1)
                pred_logits = pred_logits * pred_ious[indices]

                all_pred_boxes.append(pred_boxes[indices, torch.argmax(pred_logits, dim=1)])
                all_pred_scores.append(pred_logits.max(dim=1).values)

        height, width = all_data[fname]['height'], all_data[fname]['width']
        scale = max(height, width) / 1024.
        pred_boxes = torch.cat(all_pred_boxes) * scale
        pred_boxes[:, [0, 2]] = pred_boxes[:, [0, 2]].clamp(0, width)
        pred_boxes[:, [1, 3]] = pred_boxes[:, [1, 3]].clamp(0, height)
        pred_scores = torch.cat(all_pred_scores)
        box_area = vision_ops.box_area(pred_boxes)
        mask = (box_area < (height * width * 0.75)) & (box_area > 10)
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]

        nms_indices = vision_ops.nms(pred_boxes, pred_scores, 0.5)
        instances = Instances((height, width))
        pred_boxes = pred_boxes[nms_indices]
        pred_scores = pred_scores[nms_indices]
        instances.pred_boxes = Boxes(pred_boxes)
        instances.scores = pred_scores
        instances.pred_classes = torch.zeros(len(pred_boxes)).cuda().long()
        prediction = {"image_id": int(fname[:-4]), "instances": instances}
        coco_evaluator.process(
            [{'file_name': fname, 'height': height, 'width': width, 'image_id': int(fname[:-4])}],
            [prediction, ])
        all_predictions[fname] = prediction
        # break
    detection_results = coco_evaluator.evaluate([int(x[:-4]) for x in image_list])
    for k in detection_results['bbox']:
        results[split + '_' + k] = detection_results['bbox'][k]

    def eval_counting(thresh):
        total_mae = 0.
        total_mse = 0.
        total_nae = 0.
        total_sre = 0.
        for i, fname in enumerate(image_list):
            num_points = len(all_data[fname]['annotations']['points'])
            err = abs(num_points - (all_predictions[fname]['instances'].scores > thresh).sum())
            total_mae += err
            total_mse += err ** 2
            total_nae += err / num_points
            total_sre += err ** 2 / num_points
        cnt = len(image_list)
        mae = float(total_mae / cnt)
        mse = float((total_mse / cnt) ** 0.5)
        nae = float(total_nae / cnt)
        sre = float((total_sre / cnt) ** 0.5)
        return mae, mse, nae, sre

    if threshold is None:
        mae, mse, nae, sre = [], [], [], []
        thresholds = np.arange(0, 1., 0.01)
        for thresh in thresholds:
            mae_, mse_, nae_, sre_ = eval_counting(thresh)
            mae.append(mae_)
            mse.append(mse_)
            nae.append(nae_)
            sre.append(sre_)
        mae, mse, nae, sre, thresh = mae[np.argmin(mae)], mse[np.argmin(mae)], nae[np.argmin(mae)], sre[np.argmin(mae)], \
            thresholds[np.argmin(mae)]
        results['THRESH'] = thresh
    else:
        mae, mse, nae, sre = eval_counting(threshold)
    results[split + '_MAE'] = mae
    results[split + '_RMSE'] = mse
    results[split + '_NAE'] = nae
    results[split + '_SRE'] = sre
    return results


for n_iter in range(1, max_iter + 1):
    cls_head.train()

    targets = [all_data[all_image_list['train'][i]] for i in torch.randint(0, len(all_image_list['train']), (bs,))]
    targets = convert_to_cuda(targets)
    features = torch.cat([t['features'] for t in targets])
    num_anchors = 256
    pos_ratios = 0.25
    anchor_boxes = []
    query_features, query_labels = [], []

    for t in targets:

        fname = t['image_id']

        annotations = t['annotations']
        gt_bboxes = annotations['boxes']
        gt_points = annotations['points']

        min_scores = 0.05
        max_points = 1000

        pred_points_score = all_data[fname]['predictions']['pred_points_score']
        mask = torch.zeros(pred_points_score.size(0))
        mask[:min(pred_points_score.size(0), max_points)] = 1
        mask[pred_points_score < min_scores] = 0
        candidate_boxes = all_data[fname]['predictions']['pred_boxes'][mask.bool()].cuda()[:, :num_masks]

        iou_scores = vision_ops.box_iou(candidate_boxes.reshape(-1, 4), gt_bboxes)
        iou_scores = iou_scores.max(dim=1).values.reshape(-1, num_masks)

        anchor_indices = torch.randint(0, candidate_boxes.size(0), (num_anchors,))
        pos_mask = (iou_scores.max(1).values > 0.5).float()
        if pos_mask.sum() > 0:
            pos_indices = torch.multinomial(pos_mask, int(min(num_anchors * pos_ratios, pos_mask.sum())))
            anchor_indices[:len(pos_indices)] = pos_indices

        cur_labels = torch.zeros(len(anchor_indices), num_masks).cuda()
        cur_labels[iou_scores[anchor_indices] > 0.5] = 1.
        query_labels.append(cur_labels)
        anchor_boxes.append(candidate_boxes[anchor_indices])
        if opts.zeroshot:
            example_features = clip_text_prompts[t['class_name']].cuda().unsqueeze(0)
        else:
            example_features = all_data[fname]['example_clip_features'][
                torch.randint(0, len(all_data[fname]['example_clip_features']), (1,))].cuda()
        query_features += [example_features, ] * len(anchor_indices)
    query_labels = torch.cat(query_labels, dim=0)

    targets_a = [all_data[all_image_list['all'][i]] for i in torch.randint(0, len(all_image_list['all']), (bs,))]
    targets_a = convert_to_cuda(targets_a)
    features_a = torch.cat([t['features'] for t in targets_a])
    clip_boxes, clip_target_features, clip_query_labels = [], [], []
    for t in targets_a:
        fname = t['image_id']

        region_boxes = all_data[fname]['predictions']['clip_regions']['boxes'].float().cuda()[:, :num_masks]
        rand_indices = torch.randint(0, len(region_boxes), (min(16, len(region_boxes)),))
        for i in rand_indices:
            iou_scores = vision_ops.box_iou(region_boxes[i], region_boxes[i])
            cur_labels = torch.zeros_like(iou_scores)
            cur_labels[iou_scores > 0.5] = 1.
            clip_query_labels.append(cur_labels)
        clip_boxes.append(region_boxes[rand_indices].cuda())
        clip_target_features += [x[0].cuda() for x in
                                 all_data[fname]['predictions']['clip_regions']['clip_embeddings'][:, :num_masks][
                                     rand_indices].split(1, dim=0)]
    clip_query_labels = torch.cat(clip_query_labels)

    with torch.autograd.set_grad_enabled(True) and torch.autocast(device_type='cuda', enabled=amp):

        cls_outs = cls_head(features, anchor_boxes, query_features)

        cls_loss = F.binary_cross_entropy_with_logits(cls_outs, query_labels, reduction='none')
        loss_mask = (query_labels >= 0).float()
        cls_loss = (cls_loss * loss_mask).sum() / (loss_mask.sum() + 1e-5)

        cls_outs2 = cls_head(features_a, clip_boxes, clip_target_features)
        cls_loss2 = F.binary_cross_entropy_with_logits(cls_outs2, clip_query_labels, reduction='none')
        loss_mask = (clip_query_labels >= 0).float()
        cls_loss2 = (cls_loss2 * loss_mask).sum() / (loss_mask.sum() + 1e-5)

        loss = cls_loss + cls_loss2 * cls_loss2_weight

        update_params = (n_iter % acc_grd_step == 0)
        loss = loss / acc_grd_step
        scaler(loss, optimizer=optimizer, update_grad=update_params)

    batch_pos_ratio = (query_labels == 1).sum() / ((query_labels == 1).sum() + (query_labels == 0).sum())
    batch_pos_ratio2 = (clip_query_labels == 1).sum() / (
            (clip_query_labels == 1).sum() + (clip_query_labels == 0).sum())
    logger.msg([cls_loss, cls_loss2, batch_pos_ratio, batch_pos_ratio2], n_iter)

    if n_iter % 1000 == 0:
        results = {}
        # set the threshold at val set
        evaluate('val', results)
        evaluate('test', results, results['THRESH'])
        logger.checkpoints(n_iter)
        logger.msg(results, n_iter)
