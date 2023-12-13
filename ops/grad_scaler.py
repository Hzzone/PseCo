# -*- coding: UTF-8 -*-
'''
@Project : ICLR2022_Codes 
@File    : grad_scaler.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/1/18 8:29 PM 
'''
import torch


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self,
                 optimizer=None,
                 amp=False,
                 clip_grad=None):
        self._scaler = torch.cuda.amp.GradScaler()
        self.clip_grad = clip_grad
        self.optimizer = optimizer
        self.amp = amp

    def __call__(self, loss, optimizer=None, clip_grad=None, parameters=None, update_grad=True, backward_kwargs={}):
        if optimizer is None:
            optimizer = self.optimizer
        if clip_grad is None:
            clip_grad = self.clip_grad
        if self.amp:
            self._scaler.scale(loss).backward(**backward_kwargs)
        else:
            loss.backward(**backward_kwargs)

        norm = None
        if update_grad:
            if self.amp:
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            if clip_grad is not None:
                if parameters is None:
                    parameters = []
                    for param_group in optimizer.param_groups:
                        parameters += param_group['params']
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                if parameters is not None:
                    norm = get_grad_norm_(parameters)
            if self.amp:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
