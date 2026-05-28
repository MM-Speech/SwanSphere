import numpy as np
import torch
import torch.nn as nn


def print_arch(model, model_name='model'):
    print(f"| {model_name} Arch: ", model)
    # num_params(model, model_name=model_name)


def num_params(model, print_out=True, model_name="model"):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
    return parameters

def get_device_of_model(model):
    return model.parameters().__next__().device

def requires_grad(model):
    if isinstance(model, torch.nn.Module):
        for p in model.parameters():
            p.requires_grad = True
    else:
        model.requires_grad = True

def not_requires_grad(model):
    if isinstance(model, torch.nn.Module):
        for p in model.parameters():
            p.requires_grad = False
    else:
        model.requires_grad = False


def unwrap_model(model):
    from torch.nn.parallel import DistributedDataParallel
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model

def freeze_by_param_name(model, freeze_patterns=None, use_regex=False):
    """
    freeze_patterns: List[str]，要匹配的模式（子串或正则）
    use_regex: 为 True 时使用正则 re.search
    
    示例：冻结所有含 "layer1" 或 "layer2" 的参数，以及所有 "conv1.weight"
    例如在 ResNet 里，参数名像 "layer1.0.conv1.weight"
    model = ...
    freeze_by_param_name(model, freeze_patterns=["layer1", "layer2", "conv1.weight"])
    """
    import re
    if freeze_patterns is None:
        freeze_patterns = []
    frozen_params = []

    for name, param in model.named_parameters():
        matched = False
        for pat in freeze_patterns:
            if use_regex:
                if re.search(pat, name):
                    matched = True
                    break
            else:
                if pat in name:  # 子串匹配
                    matched = True
                    break
        if matched:
            param.requires_grad = False
            frozen_params.append(name)
            
    return frozen_params

def freeze_by_module_name(model, freeze_modules=None, exact_match=False):
    """
    freeze_modules: List[str]，模块名（来自 model.named_modules()）
    exact_match: True 表示模块名需完全相等；False 表示子串匹配
    
    示例：冻结整个 backbone 子模块
    model = ...
    freeze_by_module_name(model, freeze_modules=["backbone"], exact_match=True)

    """
    if freeze_modules is None:
        freeze_modules = []
    frozen_modules = []

    for name, module in model.named_modules():
        def should_freeze(n):
            if exact_match:
                return n in freeze_modules
            else:
                return any(pat in n for pat in freeze_modules)

        if should_freeze(name):
            for p in module.parameters(recurse=True):
                p.requires_grad = False
            frozen_modules.append(name)
    
    return frozen_modules


