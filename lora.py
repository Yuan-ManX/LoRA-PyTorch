import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class LoRALayer():
    """
    LoRALayer 类，实现了低秩适应（LoRA）机制的基础功能。

    LoRA 通过在预训练权重的基础上添加低秩矩阵来进行微调，从而减少可训练参数的数量并加速训练过程。

    参数:
        r (int): LoRA 的秩，决定了低秩矩阵 A 和 B 的维度。
        lora_alpha (int): LoRA 的缩放因子，用于调整 LoRA 层的贡献。
        lora_dropout (float): LoRA 层中使用的 Dropout 概率。如果大于0，则应用 Dropout；否则，不使用 Dropout。
        merge_weights (bool): 是否在训练前合并 LoRA 层与预训练权重。
    """
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha

        # 如果 Dropout 概率大于0，则应用 Dropout；否则，返回恒等函数
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            # 返回输入本身
            self.lora_dropout = lambda x: x
        
        # 标记权重是否已合并
        self.merged = False
        # 是否在训练前合并权重
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    """
    Embedding 类，集成了 LoRA 机制的低秩适应嵌入层。

    该类继承自 PyTorch 的 nn.Embedding，并在其中加入了 LoRA 机制，用于微调预训练的嵌入矩阵。

    参数:
        num_embeddings (int): 词汇表的大小，即嵌入矩阵的行数。
        embedding_dim (int): 嵌入向量的维度，即嵌入矩阵的列数。
        r (int, optional): LoRA 的秩，默认为0，表示不使用 LoRA。
        lora_alpha (int, optional): LoRA 的缩放因子，默认为1。
        merge_weights (bool, optional): 是否在训练前合并权重，默认为True。
        **kwargs: 其他传递给 nn.Embedding 的参数。
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        # 初始化 LoRALayer
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        
        # 如果 LoRA 的秩大于0，则定义 LoRA 的参数
        if r > 0:
            # 初始化 LoRA 矩阵 A，形状为 (r, num_embeddings)，初始值为零
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            # 初始化 LoRA 矩阵 B，形状为 (embedding_dim, r)，使用正态分布初始化
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            # 计算缩放因子
            self.scaling = self.lora_alpha / self.r
            # 冻结预训练的嵌入矩阵，使其不可训练
            self.weight.requires_grad = False
        # 重置参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置嵌入层和 LoRA 层的参数。
        """
        # 重置父类 nn.Embedding 的参数
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # 使用正态分布初始化 LoRA 矩阵 B
            nn.init.zeros_(self.lora_A)
            # 将 LoRA 矩阵 A 初始化为零
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        """
        设置模型为训练模式或评估模式，并处理权重合并。

        参数:
            mode (bool): 如果为 True，则设置为训练模式；否则，设置为评估模式。
        """
        # 设置父类 nn.Embedding 为相应的模式
        nn.Embedding.train(self, mode)
        if mode:
            # 如果在训练模式下且需要合并权重且已经合并，则取消合并
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # 从预训练权重中减去 LoRA 部分的贡献
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            # 如果在评估模式下且需要合并权重且尚未合并，则进行合并
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # 将 LoRA 部分的贡献加到预训练权重上
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        """
        前向传播方法，应用嵌入层和 LoRA 机制。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, sequence_length)

        返回:
            Tensor: 输出嵌入向量，形状为 (batch_size, sequence_length, embedding_dim)
        """
        if self.r > 0 and not self.merged:
            # 如果使用 LoRA 且尚未合并，则执行以下步骤
            # 首先调用父类的嵌入层，获取基础嵌入向量
            result = nn.Embedding.forward(self, x)
            # 计算 LoRA 矩阵 A 的嵌入向量，形状为 (batch_size, sequence_length, r)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            # 将 LoRA 矩阵 A 的嵌入向量与 LoRA 矩阵 B 相乘，并应用缩放因子
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            # 如果不使用 LoRA 或已经合并，则直接调用父类的嵌入层
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    """
    Linear 类，集成了 LoRA 机制的低秩适应全连接层。

    该类继承自 PyTorch 的 nn.Linear，并在其中加入了 LoRA 机制，用于微调预训练的全连接层权重。

    参数:
        in_features (int): 输入特征的维度。
        out_features (int): 输出特征的维度。
        r (int, optional): LoRA 的秩，默认为0，表示不使用 LoRA。
        lora_alpha (int, optional): LoRA 的缩放因子，默认为1。
        lora_dropout (float, optional): LoRA 层中使用的 Dropout 概率，默认为0.（不使用 Dropout）。
        fan_in_fan_out (bool, optional): 如果替换的层存储权重的方式为 (fan_in, fan_out)，则设置为 True，默认为 False。
        merge_weights (bool, optional): 是否在训练前合并权重，默认为 True。
        **kwargs: 其他传递给 nn.Linear 的参数。
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # 初始化 LoRALayer
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        # 存储 fan_in_fan_out 参数
        self.fan_in_fan_out = fan_in_fan_out
        
        # 如果 LoRA 的秩大于0，则定义 LoRA 的参数
        if r > 0:
            # 初始化 LoRA 矩阵 A，形状为 (r, in_features)，初始值为零
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            # 初始化 LoRA 矩阵 B，形状为 (out_features, r)，初始值为零
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # 计算缩放因子
            self.scaling = self.lora_alpha / self.r
            # 冻结预训练的全连接层权重，使其不可训练
            self.weight.requires_grad = False

        # 重置参数
        self.reset_parameters()

        # 如果 fan_in_fan_out 为 True，则将权重矩阵转置
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        """
        重置全连接层和 LoRA 层的参数。
        """
        # 重置父类 nn.Linear 的参数
        nn.Linear.reset_parameters(self)

        if hasattr(self, 'lora_A'):
            # 使用 Kaiming 均匀初始化 LoRA 矩阵 A
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # 将 LoRA 矩阵 B 初始化为零
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """
        设置模型为训练模式或评估模式，并处理权重合并。

        参数:
            mode (bool): 如果为 True，则设置为训练模式；否则，设置为评估模式。
        """
        # 定义一个辅助函数 T，用于根据 fan_in_fan_out 参数转置权重
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        # 设置父类 nn.Linear 为相应的模式
        nn.Linear.train(self, mode)

        if mode:
            # 如果在训练模式下且需要合并权重且已经合并，则取消合并
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # 从预训练权重中减去 LoRA 部分的贡献
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            # 如果在评估模式下且需要合并权重且尚未合并，则进行合并
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # 将 LoRA 部分的贡献加到预训练权重上
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        """
        前向传播方法，应用全连接层和 LoRA 机制。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, in_features)

        返回:
            Tensor: 输出张量，形状为 (batch_size, out_features)
        """
        # 定义一个辅助函数 T，用于根据 fan_in_fan_out 参数转置权重
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if self.r > 0 and not self.merged:
            # 如果使用 LoRA 且尚未合并，则执行以下步骤
            # 首先调用父类的全连接层，获取基础输出
            result = F.linear(x, T(self.weight), bias=self.bias)      

            # 计算 LoRA 部分的输出
            # 对输入 x 进行 Dropout 处理
            # 将输入与 LoRA 矩阵 A 相乘，再与 LoRA 矩阵 B 相乘，并应用缩放因子      
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            # 如果不使用 LoRA 或已经合并，则直接调用父类的全连接层
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    """
    MergedLinear 类，集成了 LoRA 机制的低秩适应全连接层，并在训练后合并 LoRA 参数。

    该类继承自 PyTorch 的 nn.Linear，并在其中加入了 LoRA 机制。通过在训练过程中应用 LoRA，
    并在训练结束后将 LoRA 参数合并到预训练的权重矩阵中，以提高推理效率。

    参数:
        in_features (int): 输入特征的维度。
        out_features (int): 输出特征的维度。
        r (int, optional): LoRA 的秩，默认为0，表示不使用 LoRA。
        lora_alpha (int, optional): LoRA 的缩放因子，默认为1。
        lora_dropout (float, optional): LoRA 层中使用的 Dropout 概率，默认为0.（不使用 Dropout）。
        enable_lora (List[bool], optional): 指示哪些输出特征维度启用 LoRA，默认为 [False]。
        fan_in_fan_out (bool, optional): 如果替换的层存储权重的方式为 (fan_in, fan_out)，则设置为 True，默认为 False。
        merge_weights (bool, optional): 是否在训练前合并权重，默认为 True。
        **kwargs: 其他传递给 nn.Linear 的参数。
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # 初始化 LoRALayer
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        
        # 确保 enable_lora 的长度必须能够整除 out_features
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        
        # 存储 enable_lora 参数
        self.enable_lora = enable_lora
        # 存储 fan_in_fan_out 参数
        self.fan_in_fan_out = fan_in_fan_out
        
        # 如果 LoRA 的秩大于0且有任何一个输出维度启用 LoRA，则定义 LoRA 的参数
        if r > 0 and any(enable_lora):
            # 初始化 LoRA 矩阵 A，形状为 (r * sum(enable_lora), in_features)，初始值为零
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            
            # 初始化 LoRA 矩阵 B，形状为 (out_features // len(enable_lora) * sum(enable_lora), r)，初始值为零
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # 适用于分组卷积，组数为 sum(enable_lora)

            # 计算缩放因子
            self.scaling = self.lora_alpha / self.r
            
            # 冻结预训练的全连接层权重，使其不可训练
            self.weight.requires_grad = False
            
            # 计算 LoRA 应用的索引
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)

        # 重置参数
        self.reset_parameters()

        # 如果 fan_in_fan_out 为 True，则将权重矩阵转置
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        """
        重置全连接层和 LoRA 层的参数。
        """
        # 重置父类 nn.Linear 的参数
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # 使用 Kaiming 均匀初始化 LoRA 矩阵 A
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # 将 LoRA 矩阵 B 初始化为零
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        """
        对输入张量进行零填充，以便与 enable_lora 对应的位置对齐。

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 填充后的张量，形状为 (out_features, ...)
        """
        # 创建与 lora_ind 对应的零张量
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        # 将输入张量填充到对应的位置
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        """
        合并 LoRA 矩阵 A 和 B，得到权重增量。

        返回:
            Tensor: 合并后的权重增量，形状为 (out_features, in_features)
        """
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        # 使用分组卷积合并 LoRA 矩阵 A 和 B
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        # 转置并填充
        return T(self.zero_pad(delta_w))  

    def train(self, mode: bool = True):
        """
        设置模型为训练模式或评估模式，并处理权重合并。

        参数:
            mode (bool): 如果为 True，则设置为训练模式；否则，设置为评估模式。
        """
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        # 设置父类 nn.Linear 为相应的模式
        nn.Linear.train(self, mode)
        if mode:
            # 如果在训练模式下且需要合并权重且已经合并，则取消合并
            if self.merge_weights and self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            # 如果在评估模式下且需要合并权重且尚未合并，则进行合并
            if self.merge_weights and not self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        """
        前向传播方法，应用全连接层和 LoRA 机制。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, in_features)

        返回:
            Tensor: 输出张量，形状为 (batch_size, out_features)
        """
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            # 如果已经合并，则直接应用全连接层
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            # 如果尚未合并，则应用全连接层并添加 LoRA 部分的贡献
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                # 计算 LoRA 部分的输出
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result


class ConvLoRA(nn.Module, LoRALayer):
    """
    ConvLoRA 类，集成了 LoRA 机制的低秩适应卷积层。

    该类继承自 PyTorch 的 nn.Module 和 LoRALayer，并在其中加入了 LoRA 机制，用于微调预训练的卷积层权重。
    通过在卷积层的权重上添加低秩矩阵，实现高效的参数微调。

    参数:
        conv_module (nn.Module): 卷积模块，如 nn.Conv2d、nn.Conv1d 或 nn.Conv3d。
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int 或 tuple): 卷积核的大小。
        r (int, optional): LoRA 的秩，默认为0，表示不使用 LoRA。
        lora_alpha (int, optional): LoRA 的缩放因子，默认为1。
        lora_dropout (float, optional): LoRA 层中使用的 Dropout 概率，默认为0.（不使用 Dropout）。
        merge_weights (bool, optional): 是否在训练前合并权重，默认为 True。
        **kwargs: 其他传递给卷积模块的参数。
    """
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()

        # 初始化卷积模块，并注册其参数
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        for name, param in self.conv.named_parameters():
            self.register_parameter(name, param)

        # 初始化 LoRALayer
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        
        # 如果 LoRA 的秩大于0，则定义 LoRA 的参数
        if r > 0:
            # 初始化 LoRA 矩阵 A，形状为 (r * kernel_size, in_channels * kernel_size)，初始值为零
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )

            # 初始化 LoRA 矩阵 B，形状为 (out_channels // groups * kernel_size, r * kernel_size)，初始值为零
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )

            # 计算缩放因子
            self.scaling = self.lora_alpha / self.r
            
            # 冻结预训练的卷积层权重，使其不可训练
            self.conv.weight.requires_grad = False
        
        # 重置参数
        self.reset_parameters()
        # 标记权重是否已合并，初始为未合并
        self.merged = False

    def reset_parameters(self):
        """
        重置卷积层和 LoRA 层的参数。
        """
        # 重置卷积层的参数
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # 使用 Kaiming 均匀初始化 LoRA 矩阵 A
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # 将 LoRA 矩阵 B 初始化为零
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        """
        设置模型为训练模式或评估模式，并处理权重合并。

        参数:
            mode (bool): 如果为 True，则设置为训练模式；否则，设置为评估模式。
        """
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # 确保权重未合并
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # 合并权重并标记
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        """
        前向传播方法，应用卷积层和 LoRA 机制。

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 输出张量
        """
        if self.r > 0 and not self.merged:
            # 如果使用 LoRA 且尚未合并，则应用 LoRA 调整后的卷积权重
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        # 否则，直接应用卷积层
        return self.conv(x)


class Conv2d(ConvLoRA):
    """
    Conv2d 类，集成了 LoRA 机制的二维卷积层。

    该类继承自 ConvLoRA，并使用 nn.Conv2d 作为基础卷积模块。
    """
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    """
    Conv1d 类，集成了 LoRA机制的一维卷积层。

    该类继承自 ConvLoRA，并使用 nn.Conv1d 作为基础卷积模块。
    """
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)
        

class Conv3d(ConvLoRA):
    """
    Conv3d 类，集成了 LoRA机制的三维卷积层。

    该类继承自 ConvLoRA，并使用 nn.Conv3d 作为基础卷积模块。
    """
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)
