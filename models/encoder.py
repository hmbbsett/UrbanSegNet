import torch
import torch.nn as nn
import math
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
try:
    from .dinov3.hub.backbones import dinov3_vitl16
except ImportError as e1:
    try:
        from dinov3.hub.backbones import dinov3_vitl16
    except ImportError as e2:
        print(f'\n❌ [严重错误] 无法导入 DINOv3。请确认 main/models/dinov3 文件夹完整。')
        raise e2

class LoRA_Layer(nn.Module):

    def __init__(self, qkv: nn.Linear, r: int=4):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        self.qkv.weight.requires_grad = False
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad = False
        self.w_a_q = nn.Linear(self.dim, r, bias=False)
        self.w_b_q = nn.Linear(r, self.dim, bias=False)
        self.w_a_v = nn.Linear(self.dim, r, bias=False)
        self.w_b_v = nn.Linear(r, self.dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b_q.weight)
        nn.init.kaiming_uniform_(self.w_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b_v.weight)

    def forward(self, x):
        base_qkv = self.qkv(x)
        delta_q = self.w_b_q(self.w_a_q(x))
        delta_v = self.w_b_v(self.w_a_v(x))
        base_qkv[:, :, :self.dim] += delta_q
        base_qkv[:, :, -self.dim:] += delta_v
        return base_qkv

class LocalDINOv3Encoder(nn.Module):

    def __init__(self, ckpt_path, img_size=1024, use_lora=True, lora_rank=4):
        super().__init__()
        self.backbone = dinov3_vitl16(pretrained=False)
        self.embed_dim = 1024
        self.patch_size = 16
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'teacher' in checkpoint:
                state_dict = checkpoint['teacher']
            else:
                state_dict = checkpoint
            clean_dict = {}
            for k, v in state_dict.items():
                k = k.replace('backbone.', '').replace('module.', '')
                clean_dict[k] = v
            if 'pos_embed' in clean_dict:
                model_shape = self.backbone.pos_embed.shape
                ckpt_shape = clean_dict['pos_embed'].shape
                if model_shape != ckpt_shape:
                    del clean_dict['pos_embed']
            msg = self.backbone.load_state_dict(clean_dict, strict=False)
        else:
            print(f'❌ [Warning] 权重文件未找到: {ckpt_path}')
        for param in self.backbone.parameters():
            param.requires_grad = False
        if use_lora:
            for block in self.backbone.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                    original_qkv = block.attn.qkv
                    block.attn.qkv = LoRA_Layer(original_qkv, r=lora_rank)

    def forward(self, x):
        out = self.backbone.forward_features(x)
        x_patch = out['x_norm_patchtokens']
        B, N, C = x_patch.shape
        H, W = (x.shape[2], x.shape[3])
        h_feat = H // self.patch_size
        w_feat = W // self.patch_size
        if N > h_feat * w_feat:
            x_patch = x_patch[:, :h_feat * w_feat, :]
        x_patch = x_patch.transpose(1, 2).reshape(B, C, h_feat, w_feat)
        return x_patch

class ResNetEncoder(nn.Module):

    def __init__(self, img_size=1024, variant='resnet18', pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
        if variant == 'resnet18':
            builder, weights, self.embed_dim = (resnet18, ResNet18_Weights.IMAGENET1K_V1, 256)
        elif variant == 'resnet50':
            builder, weights, self.embed_dim = (resnet50, ResNet50_Weights.IMAGENET1K_V2, 1024)
        else:
            raise ValueError(f'Unknown ResNet variant: {variant}')
        try:
            self.backbone = builder(weights=weights if pretrained else None)
        except Exception as e:
            raise RuntimeError(f'[ResNetEncoder] 加载 {variant} 的 ImageNet 预训练权重失败：{e}\n  这条消融的结论是"换骨干"，必须是预训练 ResNet；随机权重会让整行失去意义。\n  离线环境请先确认 torch hub 缓存（~/.cache/torch/hub/checkpoints）里已有权重，或用 TORCH_HOME 指向已缓存目录。') from e
        self.patch_size = 16

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        return x
