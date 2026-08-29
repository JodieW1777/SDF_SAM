# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
# from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder

from .mask_decoder import SDFDecoder
from .prompt_encoder import PositionEmbedding3D

from typing import List, Optional, Tuple

# 补充全局缺失导入（原生文件新增）
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

class Sam(nn.Module):
    # 改写初始化函数：替换mask_decoder为sdf_decoder
    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        sdf_decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.sdf_decoder = sdf_decoder  # 核心替换：废弃MaskDecoder
        # self.slice_fusion = CrossSliceTransformer(
        #     dim=self.image_encoder.embed_dim,
        #     depth=2,
        #     heads=8
        # )
    # 完全重写forward前向传播（核心改造）
    def forward(
            self,
            xy_slices, xz_slices, yz_slices,
            xy_rel, xz_rel, yz_rel,
            xy_boxes, xz_boxes, yz_boxes,
            query_points,
            points_per_slice,
            eikonal_mode=False,
    ) -> torch.Tensor:
    # 处理每组切片维度兼容：4维单病例自动加batch维
        def encode_plane(plane_img):
            if plane_img.dim() == 4:
                B = 1
                plane_img = plane_img.unsqueeze(0)
            else:
                B = plane_img.shape[0]
            B, N, C, H, W = plane_img.shape
            # PatchEmbed uses non-overlapping 16x16 patches. Pad only the
            # bottom/right edge so every native pixel is represented; unlike
            # resize this does not move or interpolate any image content.
            patch_h, patch_w = self.image_encoder.patch_embed.proj.kernel_size
            pad_h = (-H) % patch_h
            pad_w = (-W) % patch_w
            flat = plane_img.reshape(-1, 3, H, W)
            if pad_h or pad_w:
                flat = F.pad(flat, (0, pad_w, 0, pad_h))
            feat = self.image_encoder(flat)
            return feat.unflatten(0, (B, N)), (H + pad_h, W + pad_w)

        # Native plane sizes before the ViT. For a volume laid out as D,H,W:
        # XY is HxW, XZ is DxW and YZ is DxH.
        xy_hw = xy_slices.shape[-2:]
        xz_hw = xz_slices.shape[-2:]
        yz_hw = yz_slices.shape[-2:]
        H, W = xy_hw
        D = xz_hw[0]
        if xz_hw[1] != W or yz_hw[0] != D or yz_hw[1] != H:
            raise ValueError(
                f"Inconsistent native plane sizes: XY={xy_hw}, XZ={xz_hw}, YZ={yz_hw}"
            )

        # 三组原生尺寸图像分别过ViT图像编码器
        feat_xy, xy_padded_hw = encode_plane(xy_slices)
        feat_xz, xz_padded_hw = encode_plane(xz_slices)
        feat_yz, yz_padded_hw = encode_plane(yz_slices)

        B = feat_xy.shape[0]

        # TriPlaneSampler uses (x, y, z), hence the order must be W, H, D.
        padded_h, padded_w = xy_padded_hw
        padded_d = xz_padded_hw[0]
        if xz_padded_hw[1] != padded_w or yz_padded_hw != (padded_d, padded_h):
            raise ValueError(
                "The padded XY/XZ/YZ planes do not describe one consistent volume: "
                f"XY={xy_padded_hw}, XZ={xz_padded_hw}, YZ={yz_padded_hw}"
            )
        volume_size = torch.tensor(
            [padded_w, padded_h, padded_d], device=xy_slices.device
        ).expand(B, -1)
        Nxy = feat_xy.shape[1]
        Nxz = feat_xz.shape[1]
        Nyz = feat_yz.shape[1]

        # Dense prompt PE must match each native feature grid (the three planes
        # are generally rectangular and have different shapes).
        def dense_pe_for(feat, N):
            pe = self.prompt_encoder.pe_layer(feat.shape[-2:]).unsqueeze(0)
            return pe.repeat(B, N, 1, 1, 1)

        pe_xy = dense_pe_for(feat_xy, Nxy)
        pe_xz = dense_pe_for(feat_xz, Nxz)
        pe_yz = dense_pe_for(feat_yz, Nyz)

        # 分别为三个视角生成box稀疏prompt
        def build_prompt(box_batch, N, native_hw, padded_hw):
            sp_list = []
            native_h, native_w = native_hw
            padded_h, padded_w = padded_hw
            # Convert normalized native-image boxes into the normalized padded
            # frame used by PatchEmbed.
            native_to_padded = box_batch.new_tensor([
                native_w / padded_w,
                native_h / padded_h,
                native_w / padded_w,
                native_h / padded_h,
            ])
            for i in range(N):
                # Dataset boxes are normalized to [0,1]. PromptEncoder expects
                # coordinates in its configured input pixel frame.
                scale = box_batch.new_tensor([
                    self.prompt_encoder.input_image_size[1],
                    self.prompt_encoder.input_image_size[0],
                    self.prompt_encoder.input_image_size[1],
                    self.prompt_encoder.input_image_size[0],
                ])
                box = box_batch[:, i, :] * native_to_padded * scale
                sp, _ = self.prompt_encoder(points=None, boxes=box, masks=None)
                sp_list.append(sp)
            return torch.stack(sp_list, dim=1)

        prompt_xy = build_prompt(xy_boxes, Nxy, xy_hw, xy_padded_hw)
        prompt_xz = build_prompt(xz_boxes, Nxz, xz_hw, xz_padded_hw)
        prompt_yz = build_prompt(yz_boxes, Nyz, yz_hw, yz_padded_hw)

        # 送入三平面SDF解码器，不再传slice_z_positions

        sdf_pred = self.sdf_decoder(
            feat_xy=feat_xy,
            feat_xz=feat_xz,
            feat_yz=feat_yz,
            pe_xy=pe_xy,
            pe_xz=pe_xz,
            pe_yz=pe_yz,
            xy_rel=xy_rel,
            xz_rel=xz_rel,
            yz_rel=yz_rel,
            prompt_xy=prompt_xy,
            prompt_xz=prompt_xz,
            prompt_yz=prompt_yz,
            query_points=query_points,
            img_size=volume_size,
            # Eikonal loss differentiates the prediction w.r.t. query points
            # with create_graph=True.  Forward the caller's flag so the decoder
            # can detach grid-sampled image features and avoid requesting an
            # unsupported second derivative of grid_sample.
            eikonal_mode=eikonal_mode
        )
        return sdf_pred



    # def postprocess_masks(
    #     self,
    #     masks: torch.Tensor,
    #     input_size: Tuple[int, ...],
    #     original_size: Tuple[int, ...],
    # ) -> torch.Tensor:
    #     """
    #     Remove padding and upscale masks to the original image size.
    #
    #     Arguments:
    #       masks (torch.Tensor): Batched masks from the mask_decoder,
    #         in BxCxHxW format.
    #       input_size (tuple(int, int)): The size of the image input to the
    #         model, in (H, W) format. Used to remove padding.
    #       original_size (tuple(int, int)): The original size of the image
    #         before resizing for input to the model, in (H, W) format.
    #
    #     Returns:
    #       (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
    #         is given by original_size.
    #     """
    #     masks = F.interpolate(
    #         masks,
    #         (self.image_encoder.img_size, self.image_encoder.img_size),
    #         mode="bilinear",
    #         align_corners=False,
    #     )
    #     masks = masks[..., : input_size[0], : input_size[1]]
    #     masks = F.interpolate(
    #         masks, original_size, mode="bilinear", align_corners=False
    #     )
    #     return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

# class CrossSliceTransformer(nn.Module):
#     def __init__(self, dim, depth=2, heads=8):
#         super().__init__()
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim,
#             nhead=heads,
#             dim_feedforward=dim * 4,
#             batch_first=True,
#             norm_first=True
#         )
#
#         self.transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=depth
#         )
#
#         self.pos_emb = nn.Embedding(512, dim)
#
#     def forward(self, x):
#         """
#         x: (B, N, C, H, W)
#         """
#
#         B, N, C, H, W = x.shape
#
#         # 1. spatial pooling (你也可以换成attention pooling)
#         x = x.mean(dim=(-1, -2))   # (B, N, C)
#
#         # 2. slice position encoding
#         pos = torch.arange(N, device=x.device)
#         x = x + self.pos_emb(pos)[None, :, :]
#
#         # 3. transformer
#         x = self.transformer(x)  # (B, N, C)
#
#         # 4. restore spatial broadcast
#         x = x[:, :, :, None, None].expand(-1, -1, -1, H, W)
#
#         return x
