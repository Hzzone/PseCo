import math

import torchvision.transforms
from torch import nn
import torch
from typing import Union
import torch.distributed as dist
import torch.nn.functional as F
import collections.abc as container_abcs
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def plot_results(image, masks: torch.Tensor = None, points: torch.Tensor = None, bboxes: torch.Tensor = None,
                 show_num=False, alpha=0.35):
    plt.imshow(image)

    if masks is not None:
        masks = F.interpolate(masks, size=np.array(image).shape[:2])
        areas = (masks > 0.5).float().squeeze().sum(dim=(-1, -2))
        indices = torch.argsort(areas)
        masks = masks[indices]
        if points is not None:
            points = points[indices]
        if bboxes is not None:
            bboxes = bboxes[indices]
        for i in range(len(masks)):
            m = masks[i].squeeze().cpu().numpy()
            img = np.ones((*m.shape, 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i] * m
                plt.imshow(np.dstack((img, m * alpha)))

    if bboxes is not None:
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i].cpu().numpy()
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', lw=1, facecolor=(0, 0, 0, 0), alpha=alpha))
            if show_num and points is None:
                plt.text(x1, y1, str(i), )

    if points is not None:
        points = points.cpu().numpy()
        for i in range(len(points)):
            plt.scatter(points[i, 0], points[i, 1], s=2)
            if show_num:
                texts = np.array([str(j) for j in range(len(points))])
                if masks is not None:
                    texts = texts[indices.cpu().numpy()]
                plt.text(points[i, 0], points[i, 1], texts[i], )


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return denormalize(tensor, self.mean, self.std)


def denormalize(tensor, mean, std):
    tensor = tensor.clone()
    mean = torch.Tensor(mean).to(tensor.device)[:, None, None]
    std = torch.Tensor(std).to(tensor.device)[:, None, None]
    if tensor.ndim == 4:
        mean.unsqueeze_(0)
        std.unsqueeze_(0)
    return tensor * std + mean




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is not None:
        rt /= world_size
    return rt


def convert_to_ddp(modules: Union[list, nn.Module], **kwargs):
    if modules is None:
        return modules

    if isinstance(modules, list):
        modules = [x.cuda() for x in modules]
    else:
        modules = modules.cuda()
    if dist.is_initialized():
        device = torch.cuda.current_device()
        if isinstance(modules, list):
            modules = [torch.nn.parallel.DistributedDataParallel(x,
                                                                 device_ids=[device, ],
                                                                 output_device=device,
                                                                 **kwargs) for
                       x in modules]
        else:
            modules = torch.nn.parallel.DistributedDataParallel(modules,
                                                                device_ids=[device, ],
                                                                output_device=device,
                                                                **kwargs)

    else:
        modules = torch.nn.DataParallel(modules)

    return modules


def convert_to_cuda(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data
        return data.cuda(non_blocking=True)
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(convert_to_cuda(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, str):
        return [convert_to_cuda(d) for d in data]
    else:
        return data


def concat_all_gather(tensor):
    dtype = tensor.dtype
    tensor = tensor.float()
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    output = output.to(dtype)
    return output
