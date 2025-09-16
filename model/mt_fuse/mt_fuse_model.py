import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from model.transformer.vit_model import Attention, Mlp, Block, PatchEmbed

class CrossAttentionBlock(nn.Module):
    """跨分辨率注意力模块"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # 跨分辨率注意力
        x = x + self.attn(self.norm1(x))
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义QKV投影
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # 输出投影
        self.proj = nn.Linear(dim, dim)

        # Dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # 分离10x和4x特征
        cls_token = x[:, 0:1]
        x10x = x[:, 1:1+self.num_patches]
        x4x = x[:, 1+self.num_patches:]

        # 生成查询(10x分支)
        q = self.q(x10x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 生成键值对(4x分支)
        kv = self.kv(x4x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力加权
        x10x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x10x = self.proj(x10x)
        x10x = self.proj_drop(x10x)

        # 合并特征 (保留原始4x信息)
        x = torch.cat([cls_token, x10x, x4x], dim=1)
        return x


class MT_Fuse_Model(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, is_fuse=True):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(MT_Fuse_Model, self).__init__()
        self.is_fuse = is_fuse
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        if self.is_fuse:
            self.pos_embed = nn.Parameter(torch.zeros(1, 2*num_patches + self.num_tokens, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_ratio)

        # # 跨尺度交互模块
        # self.cross_attn_blocks = nn.ModuleList([
        #     CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias,
        #                         drop_ratio, attn_drop_ratio) for _ in range(3)
        # ])

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_ratio)
        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        if self.is_fuse:
            x10x, x4x = x[:,:3,...], x[:,3:,...]
            # [B, C, H, W] -> [B, num_patches, embed_dim]
            x10x = self.patch_embed(x10x)  # [B, 196, 768]
            x4x = self.patch_embed(x4x)
            # [1, 1, 768] -> [B, 1, 768]
            cls_token = self.cls_token.expand(x10x.shape[0], -1, -1)
            if self.dist_token is None:
                x = torch.cat((cls_token, x10x, x4x), dim=1)  # [B, 197, 768]
            else:
                # x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
                exit(-1)
        else:
            x = self.patch_embed(x)  # [B, 196, 768]
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)

            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head_drop(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)





if __name__ == '__main__':
    device = torch.device('cuda')

    model = MT_Fuse_Model(num_classes=2,embed_dim=384, num_heads=6).to(device)
    input4x = torch.randn(4, 3, 224, 224).to(device)
    input10x =torch.randn(4, 3, 224, 224).to(device)

    target = torch.tensor([1,1,0,1]).to(device)

    output = model(input10x, input4x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output,target)

    grad = torch.autograd.grad(loss, model.parameters())
    print('nihao')
