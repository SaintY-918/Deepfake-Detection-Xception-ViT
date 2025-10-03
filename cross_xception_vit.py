
import torch
import torch.nn as nn
import timm
from einops import repeat
from einops.layers.torch import Rearrange
from vit_pytorch.vit import Transformer

class CrossXceptionViT(nn.Module):
    """
    結合 XceptionNet 和 Vision Transformer 的模型。
    """
    def __init__(self, config, pretrained=True):
        super().__init__()
        
        # --- 讀取設定 ---
        dim = config['vit']['dim']
        depth = config['vit']['depth']
        heads = config['vit']['heads']
        mlp_dim = config['vit']['mlp-dim']
        dim_head = config['vit']['dim_head']
        num_classes = config['model']['num-classes']

        # --- 1. CNN 骨幹 (XceptionNet) ---
        self.cnn_backbone = timm.create_model(
            'xception', 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(-1,)
        )
        
        # ******** 核心修正：將索引 [0] 改為 [-1] ********
        # 這樣才能正確取得 XceptionNet 最後一層輸出的頻道數 (2048)
        cnn_output_channels = self.cnn_backbone.feature_info[-1]['num_chs']

        # --- 2. Patch Embedding 層 ---
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(cnn_output_channels, dim), # 現在會正確初始化為 nn.Linear(2048, 1024)
            nn.LayerNorm(dim)
        )

        # --- 3. Vision Transformer ---
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # --- 4. 分類頭 ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        feature_map = self.cnn_backbone(img)[0]
        x = self.to_patch_embedding(feature_map)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = x[:, 0]
        return self.classifier(x)