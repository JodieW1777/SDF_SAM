# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        # 生成与图像嵌入（image embedding）空间尺寸匹配的密集型位置编码 PositionEmbeddingRandom
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel  将框的坐标偏移 0.5 个像素，把坐标从 “像素的左上角” 偏移到 “像素的中心”。
        coords = boxes.reshape(-1, 2, 2)#重塑框的坐标形状，从 [B, 4] 转为 [B, 2, 2]
        # 原形状[B, 4]：每个框是(x1, y1, x2, y2)；
        # 新形状[B, 2, 2]：每个框拆分为两个角点[(x1, y1), (x2, y2)]，其中：
        # 第一维2：代表 “左上角” 和 “右下角” 两个角点；
        # 第二维2：代表每个角点的x / y坐标。
        corner_embedding = self.pe_layer.forward_with_coords(#对框的两个角点进行位置编码PositionEmbeddingRandom
            coords, self.input_image_size
        )
        # 为两个角点的位置编码添加可学习的角点专属嵌入权重
        # 通过加法将 “位置编码” 和 “角点专属语义” 融合，让模型能区分 “框的两个角点” 与普通点（如正 / 负点）的差异。
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding#返回融合了位置编码和角点语义的框嵌入向量

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:#张量的第 0 维（shape[0]）通常表示批量维度：[B, N, 2]（B=batch size，N = 点数量，2=xy 坐标）
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):#如点、框、掩码的位置）生成位置编码，基于随机高斯矩阵的位置编码（替代传统的固定正弦 / 余弦位置编码）
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()#生成一个固定的随机高斯矩阵（不训练），作为位置编码的 “基”；
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:#对归一化到 [0,1] 范围的二维坐标进行位置编码，维度为 d_1 × ... × d_n × 2 的张量
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix#与随机高斯矩阵相乘
        coords = 2 * np.pi * coords#乘以 2π
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)#正弦 / 余弦编码与拼接

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:#用于给图像 / 特征网格中的每个像素位置赋予唯一的位置特征
        """Generate positional encoding for a grid of the specified size."""
        h, w = size # 解包输入的网格尺寸：height(高)、width(宽)
        device: Any = self.positional_encoding_gaussian_matrix.device# 获取预先生成的随机矩阵所在设备（CPU/GPU）
        # 1. 创建h×w的全1张量，作为网格基础（每个位置初始值为1）
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        # 2. 计算y轴（垂直方向）的位置坐标：按行累加后减0.5，将坐标中心移到像素中心
        y_embed = grid.cumsum(dim=0) - 0.5
        # 3. 计算x轴（水平方向）的位置坐标：按列累加后减0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        # 4. 归一化坐标到[0,1]区间（消除绝对尺寸影响，适配不同大小的网格）
        y_embed = y_embed / h
        x_embed = x_embed / w

        # 5. 拼接x/y坐标，形成每个像素的(x,y)二维坐标（形状：h×w×2），然后编码
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        # 6. 维度置换：从 (H, W, C) 转为 (C, H, W)，符合PyTorch的张量格式（通道在前）
        return pe.permute(2, 0, 1)  # C x H x W



    def forward_with_coords(#对未归一化到 [0,1] 范围的坐标点进行位置编码 coords_input：B x N x 2
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        # 1. 克隆输入坐标，避免修改原张量
        coords = coords_input.clone()
        # 2. 归一化x坐标（宽度维度）：将像素坐标除以图像宽度（image_size[1]）
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        # 3. 归一化y坐标（高度维度）：将像素坐标除以图像高度（image_size[0]）
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        # 4. 转换为浮点型后，调用_pe_encoding完成最终的位置编码
        return self._pe_encoding(coords.to(torch.float))  # B x N x C



class PositionEmbedding3D(nn.Module):# 给任意3D点P=(x,y,z)生成位置编码F_3d(P)
    def __init__(self, embed_dim: int = 128, max_z: float = 100.0):
        super().__init__()
        # 复用原SAM的2D位置编码层（PositionEmbeddingRandom）处理x,y
        self.pe_2d = PositionEmbeddingRandom(embed_dim)
        # 新增z轴的位置编码（用MLP处理z坐标）
        self.z_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 2)
        ) # 拼接x,y,z编码后，映射到和SAM embed_dim一致的维度（默认256）
        self.out_proj = nn.Linear(embed_dim * 4, embed_dim * 2)

    def forward(self, points_3d: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            points_3d: [B, N_points, 3] → (x,y,z)，x,y是图像平面坐标，z是切片方向坐标
            img_size: (H,W) 对应切片的分辨率
        Returns:
            F_3d: [B, N_points, embed_dim]
        """
        B, N, _ = points_3d.shape
        # 1. 处理x,y（2D平面坐标）
        xy = points_3d[..., :2]  # [B, N, 2]
        # 用原SAM的forward_with_coords生成2D位置编码
        pe_2d = self.pe_2d.forward_with_coords(xy, img_size)  # [B, N, embed_dim*2]
        # 2. 处理z（切片方向坐标）
        z = points_3d[..., 2:]  # [B, N, 1]
        pe_z = self.z_mlp(z)  # [B, N, embed_dim*2]
        # 3. 拼接并映射到目标维度
        pe_combined = torch.cat([pe_2d, pe_z], dim=-1)  # [B, N, embed_dim*4]
        F_3d = self.out_proj(pe_combined)  # [B, N, embed_dim*2] → 即SAM的embed_dim
        return F_3d