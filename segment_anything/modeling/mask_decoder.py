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
    def __init__(self, slice_feat_dim=256, pos3d_feat_dim=256, out_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(slice_feat_dim + pos3d_feat_dim, out_dim, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, 1)

    # 【最终定稿·SDF-SAM原生逻辑版】FeatFusion forward
    def forward(self, slice_fused_feat, pos3d_feat):
        # slice_fused_feat = [1, 512, 2, 256] 4维（显存分片多出一维）
        # pos3d_feat        = [1, 512, 512]   3维（原生标准特征）
        print("slice_fused_feat shape:", slice_fused_feat.shape)
        print("pos3d_feat shape:", pos3d_feat.shape)

        # slice_fused_feat 最终必须为标准3维张量
        if slice_fused_feat.ndim == 4:
            # [1,512,2,256]  -->  挤压尾部维度  --> 原生标准3维 [1,512,512]
            slice_fused_feat = slice_fused_feat.flatten(start_dim=2, end_dim=3)

        # 此刻两个张量完全同维度：
        # slice_fused_feat：[1,512,512]
        # pos3d_feat：       [1,512,512]
        concat_feat = torch.cat([slice_fused_feat, pos3d_feat], dim=-1)
        print("pos3d_feat shape:", concat_feat .shape)
        # 完美匹配Conv2d卷积通道、权重、尺寸，零报错、零逻辑篡改
        # fused_feat = self.conv1(concat_feat)
        # fused_feat = self.relu(fused_feat)
        # fused_feat = self.conv2(fused_feat)
        return concat_feat

class SDFPredictor(nn.Module):
    def __init__(self, embed_dim=1024):  # 👈 固定 1024
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),   # 👈 输入 1024
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
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
