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

    # 完全重写forward前向传播（核心改造）
    def forward(
        self,
        slices: torch.Tensor,
        query_points: torch.Tensor,
        slice_z_positions: torch.Tensor,
        points_per_slice: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes_per_slice: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        SDF-SAM前向传播：多切片编码+3D坐标融合+SDF回归
        输入：多稀疏切片、3D查询坐标、切片Z轴空间坐标、可选prompt
        输出：所有3D查询点的SDF连续预测值
        """
        # 获取批次与切片数量
        B, N_slices = slices.shape[:2]

        # 【自动计算图像尺寸，不再写死1024！！】
        # slices shape: [B, N_slices, 3, H, W]
        H, W = slices.shape[-2], slices.shape[-1]
        img_size = (H, W)

        # 1. 批量编码所有切片图像特征
        # 分片处理切片，避免单次超大张量输入编码器
        slices_reshaped = slices.reshape(-1, 3, slices.shape[-2], slices.shape[-1])
        slices_reshaped_split = torch.split(slices_reshaped, 8, dim=0)
        slices_processed_list = []
        for sub_tensor in slices_reshaped_split:
            sub_processed = self.image_encoder(sub_tensor)
            slices_processed_list.append(sub_processed)
            del sub_tensor
            torch.cuda.empty_cache()
        slices_processed = torch.cat(slices_processed_list, dim=0)

        slice_embeddings = slices_processed.reshape(B, N_slices, -1, slices_processed.shape[-2], slices_processed.shape[-1])

        # 2. 获取切片位置编码
        slice_pe = self.prompt_encoder.get_dense_pe().unsqueeze(1).repeat(B, N_slices, 1, 1, 1)

        # 3. 逐切片编码prompt特征
        sparse_prompt_embeds = []
        for i in range(N_slices):
            points = (points_per_slice[0][:, i], points_per_slice[1][:, i]) if points_per_slice else None
            boxes = boxes_per_slice[:, i] if boxes_per_slice else None
            sparse_emb, _ = self.prompt_encoder(points=points, boxes=boxes, masks=None)
            sparse_prompt_embeds.append(sparse_emb)
        sparse_prompt_embeds = torch.stack(sparse_prompt_embeds, dim=1)

        # 4. SDF解码器前向，输出3D点SDF预测值
        sdf_pred = self.sdf_decoder(
            slice_embeddings=slice_embeddings,
            slice_pe=slice_pe,
            sparse_prompt_embeds=sparse_prompt_embeds,
            query_points=query_points,
            slice_z_positions=slice_z_positions,
            img_size=img_size
        )

        return sdf_pred

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

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

