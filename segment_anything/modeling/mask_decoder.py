import torch
import torch.nn as nn
import torch.nn.functional as F
from .prompt_encoder import PositionEmbedding3D
MaskDecoder = None

# ====================== 全新模块1：多切片特征融合（修复框提示维度BUG） ======================
class SliceFeatureFusion(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, slice_feat, prompt_feat):
        """
        适配两种输入模式：
        1. 无提示/点提示：prompt_feat [B, N_slices, 256]
        2. 框提示：自动适配维度，统一输出 [B, N_slices, 256]
        """
        B, N_slices, C, H, W = slice_feat.shape
        # 图像特征扁平化：[B, N_slices, H*W, C]
        slice_feat_flat = slice_feat.flatten(3).transpose(2, 3)

        # 核心修复：统一框/点/空提示特征维度
        if len(prompt_feat.shape) == 4:
            # 框提示输出: [B, N_slices, 1, C] -> 压缩匹配维度
            prompt_feat = prompt_feat.squeeze(2)

        # 注意力融合
        fused_feat, _ = self.attn(prompt_feat, slice_feat_flat, slice_feat_flat)
        fused_feat = self.norm(fused_feat)
        return fused_feat

# ====================== 全新模块2：2D-3D跨维度特征融合 ======================
class FeatureFusion(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, slice_fused_feat, pos3d_feat):
        # 拼接2D切片特征与3D位置特征
        concat_feat = torch.cat([slice_fused_feat, pos3d_feat], dim=-1)
        return self.fc(concat_feat)

# ====================== 全新模块3：SDF回归预测头 ======================
class SDFPredictor(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出单通道连续SDF值
        )

    def forward(self, fused_feat):
        return self.mlp(fused_feat).squeeze(-1)

# ====================== 核心：SDF解码器（修复后完整可用） ======================
class SDFDecoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.slice_fusion = SliceFeatureFusion(embed_dim)
        self.pos3d_encoder = PositionEmbedding3D(embed_dim)
        self.feat_fusion = FeatureFusion(embed_dim)
        self.sdf_predictor = SDFPredictor(embed_dim)

    def forward(
        self,
        slice_embeddings,
        slice_pe,
        sparse_prompt_embeds,
        query_points,
        slice_z_positions,
        img_size
    ):
        B, N_slices = slice_embeddings.shape[:2]
        N_query = query_points.shape[1]

        # 1. 多切片特征+提示特征融合（自动适配点/框/空提示维度）
        slice_fused = self.slice_fusion(slice_embeddings, sparse_prompt_embeds)

        # 2. 生成3D查询点位置特征
        pos3d_feat = self.pos3d_encoder(query_points, img_size)  # [B, N_query, C]

        # 3. 维度适配：切片全局特征匹配所有查询点
        slice_fused = slice_fused.mean(dim=1, keepdim=True).repeat(1, N_query, 1)
        total_fused = self.feat_fusion(slice_fused, pos3d_feat)

        # 4. 回归SDF预测值
        sdf_pred = self.sdf_predictor(total_fused)
        return sdf_pred