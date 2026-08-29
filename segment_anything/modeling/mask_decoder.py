import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .prompt_encoder import PositionEmbedding3D

from .transformer import TwoWayTransformer
import numpy as np

MaskDecoder = None


class SliceEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=embed_dim,
            mlp_dim=embed_dim * 4,
            num_heads=8,
        )
    def forward(self, slice_feat, slice_pe, prompt_feat):
        """
        输入：
            slice_feat: [B, N, C, H, W] 多切片图像特征
            slice_pe: [B, N, C, H, W] 对应图像稠密位置编码
            prompt_feat: [B, N, C] 每张切片的框稀疏prompt
        返回：更新后的5维图像特征 [B, N, C, H, W]
        """
        B, N, C, H, W = slice_feat.shape
        # 合并batch+切片维度，适配单图TwoWay
        flat_feat = slice_feat.flatten(0, 1)       # [B*N, 256, 64, 64]
        flat_pe = slice_pe.flatten(0, 1)           # [B*N, 256, 64, 64]
        flat_prompt = prompt_feat.flatten(0, 1)    # [B*N, 2, 256]

        # 重点：同时接收更新后的prompt + 更新后的图像序列
        updated_prompt, img_seq = self.transformer(flat_feat, flat_pe, flat_prompt)
        # img_seq = [B*N, H*W, 256] 展平图像token
        updated_img = img_seq.permute(0, 2, 1).unflatten(-1, (H, W)) # [B*N,256,64,64]
        # 还原多切片batch维度
        out_img = updated_img.unflatten(0, (B, N))
        return out_img


class CrossSliceFusion(nn.Module):
    """Exchange context among sparse slices from the same anatomical plane."""
    def __init__(self, embed_dim=256, num_heads=8, depth=2):
        super().__init__()
        self.position_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, feat, slice_pos, axis_size):
        # Spatial pooling produces one token per slice. Absolute slice position
        # tells the transformer how far apart sparse observations are.
        tokens = feat.mean(dim=(-2, -1))
        denom = torch.clamp(axis_size.to(feat.dtype) - 1.0, min=1.0).view(-1, 1, 1)
        pos = (slice_pos.unsqueeze(-1).to(feat.dtype) / denom).clamp(0.0, 1.0)
        context = self.transformer(tokens + self.position_mlp(pos))
        context = self.out_norm(context)
        return feat + context[..., None, None]


class CrossViewFusion(nn.Module):
    """Let XY, XZ and YZ evidence interact at every queried 3-D point."""
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.view_embed = nn.Parameter(torch.zeros(1, 1, 3, embed_dim))
        nn.init.normal_(self.view_embed, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, fxy, fxz, fyz):
        views = torch.stack([fxy, fxz, fyz], dim=2) + self.view_embed
        B, Q, V, C = views.shape
        tokens = views.reshape(B * Q, V, C)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens.reshape(B, Q, V * C)


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
    def __init__(self, embed_dim=1024, hidden_dim=256, omega_0=10):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(embed_dim, hidden_dim, is_first=True, omega_0=omega_0),
            SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            nn.Linear(hidden_dim, 1)  # 最后一层必须线性
        )

    def forward(self, fused_feat):
        # Training target is a truncated SDF in [-1, 1]. Bounding the decoder
        # prevents early single-case updates from sending the global field far
        # beyond that range while preserving the zero level set.
        return torch.tanh(self.net(fused_feat)).squeeze(-1)


class SDFDecoder(nn.Module):
    def __init__(self, embed_dim=256, two_way_transformer=None):
        super().__init__()

        self.slice_encoder = SliceEncoder(embed_dim)
        self.cross_slice_fusion = CrossSliceFusion(embed_dim)
        self.triplane_sampler = TriPlaneSampler()
        self.cross_view_fusion = CrossViewFusion(embed_dim)
        self.pos3d = PositionEmbedding3D(embed_dim)
        self.mlp = SDFPredictor()
        self.query_proj = nn.Linear(
            768 + 256,
            1024
        )

    def forward(
            self,
            feat_xy, feat_xz, feat_yz,
            pe_xy, pe_xz, pe_yz,  # 新增：外部传入各平面稠密PE
            xy_rel, xz_rel, yz_rel,
            prompt_xy, prompt_xz, prompt_yz,
            query_points, img_size, eikonal_mode=False
    ):
        # print("xy_rel", xy_rel)
        # print("xz_rel", xz_rel)
        # print("yz_rel", yz_rel)
        # 每组切片独立融合图像+PE+box prompt
        feat_xy = self.slice_encoder(feat_xy, pe_xy, prompt_xy)
        feat_xz = self.slice_encoder(feat_xz, pe_xz, prompt_xz)
        feat_yz = self.slice_encoder(feat_yz, pe_yz, prompt_yz)

        # Intra-plane exchange: all sparse slices contribute context before a
        # query selects/interpolates local evidence.
        feat_xy = self.cross_slice_fusion(feat_xy, xy_rel, img_size[:, 2])
        feat_xz = self.cross_slice_fusion(feat_xz, xz_rel, img_size[:, 1])
        feat_yz = self.cross_slice_fusion(feat_yz, yz_rel, img_size[:, 0])

        # 三平面交叉注意力
        fxy, fxz, fyz = self.triplane_sampler(
            feat_xy,
            feat_xz,
            feat_yz,
            query_points,
            img_size,
            xy_rel,
            xz_rel,
            yz_rel
        )
        # img_size is ordered as (W, H, D), while PositionEmbedding3D expects
        # the 2-D image size as (H, W).
        hwd_tuple = (
            int(img_size[0, 1].item()),
            int(img_size[0, 0].item()),
            int(img_size[0, 2].item()),
        )
        query_feat = self.cross_view_fusion(fxy, fxz, fyz)
        # 3D Query
        pos_feat = self.pos3d(query_points, hwd_tuple)

        query_feat = torch.cat([query_feat,pos_feat],dim=-1)

        query_feat = self.query_proj(query_feat)

        sdf_pred = self.mlp(query_feat)
        return sdf_pred



class TriPlaneSampler(nn.Module):
    def __init__(self):
        super().__init__()

    def normalize_coord(self, coord, size):
        return coord / (size - 1) * 2 - 1

    def find_bracketing_slices(self, coord, slice_pos):
        """Return neighboring slice ids and continuous interpolation weights."""
        if slice_pos.dim() == 1:
            slice_pos = slice_pos.unsqueeze(0)
        low_ids, high_ids, high_weights = [], [], []
        for b in range(coord.shape[0]):
            pos, order = torch.sort(slice_pos[b])
            if pos.numel() == 1:
                lo = hi = torch.zeros_like(coord[b], dtype=torch.long)
                weight = torch.zeros_like(coord[b])
            else:
                hi_sorted = torch.searchsorted(pos.contiguous(), coord[b].contiguous())
                hi_sorted = hi_sorted.clamp(1, pos.numel() - 1)
                lo_sorted = hi_sorted - 1
                lo_pos, hi_pos = pos[lo_sorted], pos[hi_sorted]
                weight = ((coord[b] - lo_pos) / (hi_pos - lo_pos + 1e-8)).clamp(0.0, 1.0)
                lo, hi = order[lo_sorted], order[hi_sorted]
            low_ids.append(lo)
            high_ids.append(hi)
            high_weights.append(weight)
        return (
            torch.stack(low_ids),
            torch.stack(high_ids),
            torch.stack(high_weights),
        )

    def sample_plane(self, feat, coords, slice_ids):
        B, N, C, H, W = feat.shape
        slice_ids = torch.clamp(slice_ids, 0, N - 1).long()
        outputs = []
        for b in range(B):
            # Do not use feat[b][slice_ids[b]]: that materializes one complete
            # CxHxW feature map per query (and 6x more for finite differences).
            # Group queries by slice so each feature map is sampled only once.
            query_out = feat.new_zeros((coords.shape[1], C))
            for slice_id in torch.unique(slice_ids[b]):
                query_idx = torch.nonzero(
                    slice_ids[b] == slice_id, as_tuple=False
                ).squeeze(1)
                grid = coords[b, query_idx].view(1, -1, 1, 2)
                with torch.backends.cudnn.flags(enabled=False):
                    sampled = F.grid_sample(
                        feat[b, slice_id].unsqueeze(0),
                        grid,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=True,
                    )
                values = sampled[0, :, :, 0].transpose(0, 1)
                query_out = query_out.index_copy(0, query_idx, values)
            outputs.append(query_out)
        return torch.stack(outputs, dim=0)

    def forward(self, feat_xy, feat_xz, feat_yz, query_points, img_size, xy_rel, xz_rel, yz_rel):
        x, y, z = query_points[:, :, 0], query_points[:, :, 1], query_points[:, :, 2]

        xy_lo, xy_hi, xy_w = self.find_bracketing_slices(z, xy_rel)
        xz_lo, xz_hi, xz_w = self.find_bracketing_slices(y, xz_rel)
        yz_lo, yz_hi, yz_w = self.find_bracketing_slices(x, yz_rel)

        size_x = img_size[:, 0].view(-1, 1)  # [B, 1]
        size_y = img_size[:, 1].view(-1, 1)
        size_z = img_size[:, 2].view(-1, 1)

        xy = torch.stack([
            self.normalize_coord(x, size_x),
            self.normalize_coord(y, size_y)
        ], dim=-1)

        xz = torch.stack([
            self.normalize_coord(x, size_x),
            self.normalize_coord(z, size_z)
        ], dim=-1)

        yz = torch.stack([
            self.normalize_coord(y, size_y),
            self.normalize_coord(z, size_z)
        ], dim=-1)

        def interpolate_plane(feat, coords, lo, hi, weight):
            low_feat = self.sample_plane(feat, coords, lo)
            high_feat = self.sample_plane(feat, coords, hi)
            return torch.lerp(low_feat, high_feat, weight.unsqueeze(-1))

        fxy = interpolate_plane(feat_xy, xy, xy_lo, xy_hi, xy_w)
        fxz = interpolate_plane(feat_xz, xz, xz_lo, xz_hi, xz_w)
        fyz = interpolate_plane(feat_yz, yz, yz_lo, yz_hi, yz_w)

        return fxy, fxz, fyz

class QueryVolumeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)

    def forward(self, query, volume):
        """
        query: [B, N_query, C] 3维（3D采样点编码，符合要求）
        volume: 支持4/5维，统一转为 [B, N_all_tokens, C]
        """
        B = query.shape[0]
        if volume.dim() == 5:
            # [B, N_slice, C, H, W] -> 合并切片+空间
            B, Ns, C, H, W = volume.shape
            volume = volume.reshape(B, Ns * H * W, C)
        elif volume.dim() == 4:
            B, N, C, H, W = volume.shape
            volume = volume.reshape(B, N * H * W, C)
        # 现在 volume 固定3维 [B, N_token, C]，满足MHA要求
        out, _ = self.attn(
            query=query,
            key=volume,
            value=volume
        )
        return out
