import torch
import torch.nn as nn
import torch.nn.functional as F
from .prompt_encoder import PositionEmbedding3D
# ====================== 核心：直接导入官方原生Transformer ======================
from .transformer import TwoWayTransformer

MaskDecoder = None


class SliceFeatureFusion(nn.Module):
    """
    【正统官方编码】
    不复刻、不造轮子
    直接调用 SAM/MedSAM 原生 TwoWayTransformer 完成切片-Prompt交叉编码
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # 完全使用官方原版Transformer解码器
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=embed_dim,
            mlp_dim=embed_dim * 4,
            num_heads=8,
        )

    def forward(self, slice_feat, slice_pe, prompt_feat):
        """
        输入与官方Transformer完全对齐
        自动支持批量多切片
        """
        B, N_slices, C, H, W = slice_feat.shape

        # 合并 batch 和切片维度，骗过官方单图Transformer（完美兼容）
        # [B, N, C, H, W] -> [B*N, C, H, W]
        flat_feat = slice_feat.flatten(0, 1)
        flat_pe = slice_pe.flatten(0, 1)
        flat_prompt = prompt_feat.flatten(0, 1)

        # ======================
        # 完全原版官方编码
        # ======================
        # image_embedding, image_pe, sparse_prompt
        out_feat, _ = self.transformer(flat_feat, flat_pe, flat_prompt)

        # 还原多切片维度
        out_feat = out_feat.unflatten(0, (B, N_slices))
        return out_feat


class FeatureFusion(nn.Module):
    """2D官方原生语义特征 + 3D空间几何特征 拼接融合"""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, slice_fused_feat, pos3d_feat):
        concat_feat = torch.cat([slice_fused_feat, pos3d_feat], dim=-1)
        return self.fc(concat_feat)


class SDFPredictor(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, fused_feat):
        return self.mlp(fused_feat).squeeze(-1)


class SDFDecoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.slice_encoder = SliceFeatureFusion(embed_dim)
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

        # 1. 调用【官方原生Transformer】完成所有切片编码
        # 完全原版、完全匹配预训练、无任何复刻误差
        slice_local_feat = self.slice_encoder(slice_embeddings, slice_pe, sparse_prompt_embeds)

        # 2. 多切片局部特征 -> 全局特征聚合（自研增量创新）
        global_slice_feat = slice_local_feat.mean(dim=1, keepdim=True)
        # mask_decoder.py 中 forward 函数内，repeat 行之前
        print("global_slice_feat shape:", global_slice_feat.shape)  # 打印维度
        global_slice_feat = global_slice_feat.repeat(1, N_query, 1, 1)

        # 3. 3D位置编码 & 跨维度融合
        pos3d_feat = self.pos3d_encoder(query_points, img_size)
        total_fused = self.feat_fusion(global_slice_feat, pos3d_feat)

        # 4. SDF回归
        sdf_pred = self.sdf_predictor(total_fused)

        return sdf_pred
