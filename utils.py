import torch
import torch.nn as nn
from typing import Dict

from lora import LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    将模型中除了 LoRA 层之外的所有参数设置为不可训练，仅保留 LoRA 相关参数可训练。

    该函数遍历模型的所有参数，根据参数名称是否包含 'lora_' 来决定是否将其设置为可训练。
    同时，根据 `bias` 参数的值，决定是否将偏置项设置为可训练。

    参数:
        model (nn.Module): 需要设置的 PyTorch 模型。
        bias (str, optional): 控制偏置项是否可训练的选项。选项包括：
            - 'none': 不设置任何偏置项为可训练（默认）。
            - 'all': 将所有偏置项设置为可训练。
            - 'lora_only': 仅将属于 LoRA 层的偏置项设置为可训练。
    """
    # 遍历模型的所有参数
    for n, p in model.named_parameters():
        # 如果参数名称中不包含 'lora_'，则将其设置为不可训练
        if 'lora_' not in n:
            p.requires_grad = False
    
    # 根据 bias 参数的值处理偏置项
    if bias == 'none':
        # 不设置任何偏置项为可训练
        return
    elif bias == 'all':
        # 将所有偏置项设置为可训练
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        # 仅将属于 LoRA 层的偏置项设置为可训练
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        # 如果传入的 bias 参数值不支持，则抛出异常
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """
    获取模型中 LoRA 相关的状态字典。

    该函数从模型的整个状态字典中提取与 LoRA 相关的参数，包括 LoRA 矩阵 A、B 以及可选的偏置项。

    参数:
        model (nn.Module): 需要提取状态的 PyTorch 模型。
        bias (str, optional): 控制是否包含偏置项的选项。选项包括：
            - 'none': 仅返回 LoRA 矩阵 A 和 B（默认）。
            - 'all': 返回 LoRA 矩阵 A、B 以及所有偏置项。
            - 'lora_only': 返回 LoRA 矩阵 A、B 以及仅属于 LoRA 层的偏置项。

    返回:
        Dict[str, torch.Tensor]: 包含 LoRA 相关参数及其对应张量的字典。
    """
    # 获取模型的整个状态字典
    my_state_dict = model.state_dict()

    if bias == 'none':
        # 仅返回包含 'lora_' 的参数
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        # 返回包含 'lora_' 和 'bias' 的参数
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        # 初始化一个空的字典，用于存储要返回的参数
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                # 如果参数名称包含 'lora_'，则添加到字典中
                to_return[k] = my_state_dict[k]
                # 构建对应的偏置项名称，例如 'lora_A' -> 'bias_A'
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    # 如果对应的偏置项存在，则也添加到字典中
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        # 如果传入的 bias 参数值不支持，则抛出异常
        raise NotImplementedError
