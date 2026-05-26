import torch
import torch.nn as nn
import math
import os
import sys

# Path injection for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from .dinov3.hub.backbones import dinov3_vitl16
except ImportError as e1:
    try:
        from dinov3.hub.backbones import dinov3_vitl16
    except ImportError as e2:
        print(f"\nFailed to import DINOv3. Ensure the dinov3/ directory is present and complete.")
        raise e2


class LoRA_Layer(nn.Module):
    def __init__(self, qkv: nn.Linear, r: int = 4):
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
                k = k.replace("backbone.", "").replace("module.", "")
                clean_dict[k] = v

            if 'pos_embed' in clean_dict:
                model_shape = self.backbone.pos_embed.shape
                ckpt_shape = clean_dict['pos_embed'].shape
                if model_shape != ckpt_shape:
                    del clean_dict['pos_embed']

            msg = self.backbone.load_state_dict(clean_dict, strict=False)
        else:
            print(f"Pretrained weights not found: {ckpt_path}")

        # Inject LoRA adapters
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
        H, W = x.shape[2], x.shape[3]
        h_feat = H // self.patch_size
        w_feat = W // self.patch_size

        if N > h_feat * w_feat:
            x_patch = x_patch[:, :h_feat * w_feat, :]

        x_patch = x_patch.transpose(1, 2).reshape(B, C, h_feat, w_feat)
        return x_patch