import os
import sys

sys.path.insert(0, '/home/zzhuang/PseCo')
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
from ops.foundation_models.segment_anything.utils.amg import batched_mask_to_box
from ops.ops import _nms, plot_results, convert_to_cuda
import json

torch.autograd.set_grad_enabled(False)
torch.multiprocessing.set_sharing_strategy('file_system')
from ops.foundation_models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, \
    build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
opt = parser.parse_args()
import torch.distributed as dist

dist.init_process_group(backend='nccl', init_method='env://')

torch.cuda.set_device(dist.get_rank())


def chunks(arr, m):
    import math
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


from models import PointDecoder

project_root = '/home/zzhuang/PseCo'

sam = build_sam_vit_h().cuda().eval()
point_decoder = PointDecoder(sam).cuda().eval()
state_dict = torch.load(f'{project_root}/data/fsc147/checkpoints/point_decoder_vith.pth',
                        map_location='cpu')
point_decoder.load_state_dict(state_dict)
all_data = torch.load(f'{project_root}/data/fscd_lvis/sam/all_data_vith.pth', map_location='cpu')
base_path = f'{project_root}/data/fscd_lvis/sam/all_predictions_vith'


for fname in tqdm.tqdm(all_data):
    target = all_data[fname]
    target['image_id'] = fname
    transform = A.Compose([
        A.LongestMaxSize(256),
        A.PadIfNeeded(256, 256, border_mode=0, position=A.PadIfNeeded.PositionType.TOP_LEFT),
    ])
    mask = Image.fromarray(
        transform(image=np.ones((target['height'], target['width'])).astype(np.uint8) * 255)['image'])
    mask = np.array(mask) > 128
    target['mask'] = torch.from_numpy(mask).reshape(1, 1, 256, 256).bool().float()

all_image_list = {'train': [], 'val': [], 'test': [], 'all': []}
for fname in all_data:
    # if all_data[fname]['split'] == 'train':
    #     if (all_data[fname]['annotations']['points'].size(0) + all_data[fname]['segment_anything']['points'].size(
    #             0)) != 0:
    #         all_image_list[all_data[fname]['split']].append(fname)
    # else:
    all_image_list[all_data[fname]['split']].append(fname)
    all_image_list['all'].append(fname)
all_file_names = chunks(all_image_list['all'], dist.get_world_size())[dist.get_rank()]

save_path = f'{base_path}_{dist.get_rank()}.pth'

if os.path.exists(save_path):
    predictions = torch.load(save_path, map_location='cpu')
else:
    predictions = {}

print(dist.get_rank(), len(all_file_names))
for n_iter, fname in enumerate(tqdm.tqdm(all_file_names)):
    if fname in predictions:
        continue
    features = all_data[fname]['features'].cuda()
    with torch.no_grad():
        # point_decoder.max_points = 256
        # point_decoder.point_threshold = 0.05
        # point_decoder.nms_kernel_size = 5
        point_decoder.max_points = 2000
        point_decoder.point_threshold = 0.01
        point_decoder.nms_kernel_size = 3
        outputs_heatmaps = point_decoder(features, masks=all_data[fname]['mask'].cuda())

    pred_points = outputs_heatmaps['pred_points'].squeeze()
    pred_points_score = outputs_heatmaps['pred_points_score'].squeeze()

    all_pred_boxes = []
    all_pred_scores = []
    for indices in torch.arange(len(pred_points)).split(256):
        with torch.no_grad():
            outputs_points = sam.forward_sam_with_embeddings(features, points=pred_points[indices].reshape(-1, 2))
            pred_boxes = outputs_points['pred_boxes']
            pred_logits = outputs_points['pred_ious']

            for anchor_size in [8, ]:
                anchor = torch.Tensor([[-anchor_size, -anchor_size, anchor_size, anchor_size]]).cuda()
                anchor_boxes = pred_points[indices].reshape(-1, 2).repeat(1, 2) + anchor
                anchor_boxes = anchor_boxes.clamp(0., 1024.)
                outputs_boxes = sam.forward_sam_with_embeddings(features, points=pred_points[indices].reshape(-1, 2),
                                                                boxes=anchor_boxes)
                pred_logits = torch.cat([pred_logits, outputs_boxes['pred_ious'][:, 1].unsqueeze(1)], dim=1)
                pred_boxes = torch.cat([pred_boxes, outputs_boxes['pred_boxes'][:, 1].unsqueeze(1)], dim=1)

            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_logits)
    pred_boxes = torch.cat(all_pred_boxes).cpu()
    pred_scores = torch.cat(all_pred_scores).cpu()

    predictions[fname] = {
        'pred_boxes': pred_boxes,
        'pred_ious': pred_scores,
        'pred_points_score': pred_points_score,
        'pred_points': pred_points,
    }
    # break

    if n_iter % 100 == 0:
        torch.save(predictions, save_path)
# torch.save(predictions, save_path)

from ops.foundation_models import clip
import numpy as np

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
import torchvision

normalize = torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                             std=(0.26862954, 0.26130258, 0.27577711))


def read_image(fname):
    img = Image.open(f'{project_root}/data/fsc147/images_384_VarV2/{fname}')
    transform = A.Compose([
        A.LongestMaxSize(1024),
        A.PadIfNeeded(1024, border_mode=0, position=A.PadIfNeeded.PositionType.TOP_LEFT),
    ])
    img = Image.fromarray(transform(image=np.array(img))['image'])
    return img


def extract_clip_features(fname, bboxes):
    image = read_image(fname)
    examples = []
    for box in bboxes:
        example = image.crop(box.long().tolist())
        example = example.resize((224, 224))
        example = normalize(to_tensor(example)).unsqueeze(0)
        examples.append(example)
    examples = torch.cat(examples)
    e = []
    with torch.no_grad():
        for indices in torch.arange(len(examples)).split(256):
            e.append(model.encode_image(examples[indices].cuda()).float())
    e = torch.cat(e, dim=0)
    e = F.normalize(e, dim=1).cpu()
    return e


for n_iter, fname in enumerate(tqdm.tqdm(all_file_names)):
    if 'clip_regions' in predictions[fname]:
        continue
    print(fname)
    pred_boxes = predictions[fname]['pred_boxes'].cuda()
    pred_points_score = predictions[fname]['pred_points_score'].cuda()
    rand_indices = torch.multinomial(pred_points_score, min(len(pred_boxes), 256), replacement=False)
    boxes = pred_boxes[rand_indices]
    predictions[fname]['clip_regions'] = {
        'clip_embeddings': extract_clip_features(fname, boxes.reshape(-1, 4)).view(-1, boxes.size(1), 512),
        'boxes': boxes,
    }
    if n_iter % 100 == 0:
        torch.save(predictions, save_path)
torch.save(predictions, save_path)

predictions = {}
for i in range(dist.get_world_size()):
    data = torch.load(f'{base_path}_{i}.pth', map_location='cpu')
    for fname in data:
        predictions[fname] = data[fname]
torch.save(predictions, base_path + '.pth')

# break
