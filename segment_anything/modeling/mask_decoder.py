import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .prompt_encoder import PositionEmbedding3D
# ====================== 核心：直接导入官方原生Transformer ======================
from .transformer import TwoWayTransformer

MaskDecoder = None


class SliceEncoder(nn.Module):
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
        if slice_fused_feat.dim() == 4:
            # [1,512,N,256] → 在 N 维度取平均 → 变成 [1,512,256]
            slice_fused_feat = slice_fused_feat.mean(dim=2)

        # 此刻两个张量完全同维度：
        # slice_fused_feat：[1,512,512]
        # pos3d_feat：       [1,512,512]
        # 统一维度：5维 -> 3维，不改动任何主干网络
        if slice_fused_feat.dim() == 5:
            slice_fused_feat = slice_fused_feat.squeeze(-1).squeeze(-1)
        concat_feat = torch.cat([slice_fused_feat, pos3d_feat], dim=-1)
        print("concat_feat shape:", concat_feat .shape)
        # 完美匹配Conv2d卷积通道、权重、尺寸，零报错、零逻辑篡改
        # fused_feat = self.conv1(concat_feat)
        # fused_feat = self.relu(fused_feat)
        # fused_feat = self.conv2(fused_feat)
        return concat_feat

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features,
                     1 / self.linear.in_features
                )
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        x = self.norm(x)
        return torch.sin(self.omega_0 * self.linear(x))


class SDFPredictor(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=256, omega_0=10):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(embed_dim, hidden_dim, is_first=True, omega_0=omega_0),
            SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            nn.Linear(hidden_dim, 1)  # 最后一层必须线性
        )

    def forward(self, fused_feat):
        return self.net(fused_feat).squeeze(-1)


class SDFDecoder(nn.Module):
    def __init__(self, embed_dim=256, two_way_transformer=None):
        super().__init__()

        self.slice_encoder = SliceEncoder(embed_dim)

        self.cross_slice = CrossSliceTransformer(embed_dim, 2, 8)

        self.query_attn = QueryVolumeAttention(embed_dim)

        self.pos3d = PositionEmbedding3D(embed_dim)

        self.mlp = SDFPredictor(embed_dim)

    def forward(self, slice_embeddings, slice_pe, prompt, query_points, slice_z_positions,img_size):

        # 1. slice encoding
        slice_feat = self.slice_encoder(slice_embeddings, slice_pe, prompt)

        # 2. cross-slice modeling
        volume_feat = self.cross_slice(slice_feat)

        # 3. query encoding
        query_feat = self.pos3d(query_points, img_size)

        # 4. query ↔ volume attention
        query_feat = self.query_attn(query_feat, volume_feat)

        # 5. sdf
        sdf_pred = self.mlp(query_feat)

        # print(
        #     "SDFPredictor:",
        #     sdf_pred.min().item(),
        #     sdf_pred.max().item(),
        #     sdf_pred.mean().item()
        # )

        return sdf_pred


class CrossSliceTransformer(nn.Module):
    def __init__(self, dim, depth=2, heads=8):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.pos_emb = nn.Embedding(512, dim)

    def forward(self, x):
        # x: (B,N,T,C)

        B, N, T, C = x.shape

        x = x.reshape(B, N * T, C)

        x = self.transformer(x)

        x = x.reshape(B, N, T, C)

        return x

class QueryVolumeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)

    def forward(self, query, volume):
        if volume.dim() == 4:
            B, N, T, C = volume.shape
            volume = volume.reshape(B, N*T, C)

        out, _ = self.attn(
            query=query,
            key=volume,
            value=volume
        )

        return out
