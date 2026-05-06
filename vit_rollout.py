import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
from timm.models.vision_transformer import Attention 

def rollout(attentions, discard_ratio, head_fusion):
    if not attentions:
        raise ValueError("Attentions is empty")
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0,indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            print(attention_heads_fused.shape, I.shape)
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model 
        self.model.train()
        #print(model)
##model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                print(f"Registered forward hook for {name}")
            else:
                pass
                #print(f"Layer {name} is not an attention layer")
        self.attentions = []

    def get_attention(self, module, input, output):
        print(f"当前层：{module}")
        #print(f"钩子捕获的原始形状: {output.shape}")
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        self.model.train()
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)


class MVITAttentionRollout:
    def __init__(self, model, head_fusion="mean", discard_ratio=0.9):
        self.model = model 
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        
        # 猴子补丁：遍历模型，替换所有 Attention 模块的 forward 方法
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                self._replace_attention_forward(module, name)
                print(f"✓ 成功替换 Attention 模块: {name}")

    def _replace_attention_forward(self, module, module_name):
        """
        替换 Attention 模块的 forward 方法，在 softmax 后保存注意力权重
        参数：
            module: 要替换的 Attention 模块实例
            module_name: 模块的完整路径名称（用于日志）
        """
        # 保存原始 forward 方法，以便在新方法中调用
        original_forward = module.forward
        
        # 定义新的 forward 方法
        def new_forward(x,attn_mask=None):
            # 复制原始 Attention.forward 的计算逻辑
            B, N, C = x.shape
            # QKV 投影：(B, N, C) → (B, N, 3*C) → 拆分 Q/K/V
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Q/K/V 形状：(B, num_heads, N, C//num_heads)
            
            # 计算注意力分数：Q @ K^T，然后缩放
            attn = (q @ k.transpose(-2, -1)) * module.scale  # 形状：(B, num_heads, N, N)
            if attn_mask is not None:
            # 注意：attn_mask 的形状可能需要调整，以匹配 attn 的形状
            # 这里假设 attn_mask 已经是正确的形状，或者需要扩展到 num_heads 维度
                attn = attn + attn_mask
            # Softmax 归一化：得到注意力权重（这是我们需要捕获的）
            attn = attn.softmax(dim=-1)  # 形状：(B, num_heads, N, N)
            
            # 保存注意力权重到 self.attentions
            self.attentions.append(attn.cpu())
            
            # 继续执行原始 forward 方法的剩余逻辑
            attn = module.attn_drop(attn)  # 注意力 Dropout
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 注意力加权求和
            x = module.proj(x)  # 输出投影
            x = module.proj_drop(x)  # 输出 Dropout
            return x
        
        # 替换模块的 forward 方法
        module.forward = new_forward

    def get_attention(self, module, input, output):
        """
        旧的钩子函数，现在不再使用（猴子补丁方案直接保存注意力权重）
        """
        pass

    def __call__(self, input_tensor):
        self.attentions = []
        # 无需设置 train 模式，猴子补丁不受模式影响
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)