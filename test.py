# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import nibabel as nib
from skimage import transform
from skimage.measure import marching_cubes
import trimesh
import tempfile
from segment_anything import build_sam_sdf

from scipy.interpolate import griddata

# ===================== 全局超参（修改切片数量为4） =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024
NUM_SLICES = 4  # 改为4张切片输入
SLICE_BATCH_SIZE = 4  # 批次大小等于总切片，一次推理完成
NUM_QUERY = 3000  # 每张切片的查询点数量
# 路径配置
MODEL_WEIGHT = "best_sdf_sam.pth"
NII_PATH = "data_test/FLARE22_Tr_0002_0000.nii.gz"
# 从3D Slicer导出的像素框 [x0, y0, z0, x1, y1, z1]
GLOBAL_BOX = [70, 109, 21, 215, 218, 72]


# ===================== 工具函数 =====================
def _resize_to_1024(img_slice, target_size=1024):
    h, w = img_slice.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = transform.resize(img_slice, (new_h, new_w), preserve_range=True)
    img_pad = np.zeros((target_size, target_size), dtype=np.float32)
    offset_h = (target_size - new_h) // 2
    offset_w = (target_size - new_w) // 2
    img_pad[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = img_resized
    return img_pad


def _resize_sdf_back(sdf_pred, ori_h, ori_w):
    """将1024x1024的SDF预测结果缩放回原始切片尺寸"""
    # 先裁剪掉padding部分
    h, w = sdf_pred.shape
    max_dim = max(ori_h, ori_w)
    scale = max_dim / IMG_SIZE
    new_h, new_w = int(ori_h / scale), int(ori_w / scale)
    offset_h = (IMG_SIZE - new_h) // 2
    offset_w = (IMG_SIZE - new_w) // 2
    sdf_crop = sdf_pred[offset_h:offset_h + new_h, offset_w:offset_w + new_w]
    # 缩放回原始尺寸
    sdf_resized = transform.resize(sdf_crop, (ori_h, ori_w), preserve_range=True)
    return sdf_resized


# ===================== 1. 仅在框范围内采样4张有效切片 =====================
def get_valid_slices_in_box(img_vol, bbox, H, W):
    """
    从3D框的Z轴覆盖范围内采样，自动剔除Z区间首尾边缘，只取中间切片
    :param img_vol: 3D体数据 (D, H, W)
    :param bbox: 3D像素框 [x0, y0, z0, x1, y1, z1]
    :param H: 原始高度
    :param W: 原始宽度
    :return: 采样的切片索引
    """
    if len(bbox) != 6:
        raise ValueError(f"3D框坐标需为6个值 [x0,y0,z0,x1,y1,z1]，当前输入：{bbox}")
    x0, y0, z0, x1, y1, z1 = bbox
    # 修正XY越界
    x0 = max(0, x0)
    y0 = max(0, y0)
    z0 = max(0, z0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    z1 = min(img_vol.shape[0], z1)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        raise ValueError(f"无效的3D框坐标：{bbox}，x:{x0}-{x1}, y:{y0}-{y1}, z:{z0}-{z1}")

    # 框内全部z索引
    all_z = np.arange(z0, z1 + 1)
    total_z = len(all_z)
    # 裁剪前后10%，剔除头尾薄层
    cut_ratio = 0.1
    start_cut = int(total_z * cut_ratio)
    end_cut = int(total_z * (1 - cut_ratio))
    # 防止裁剪后区间不足4张，兜底不裁剪
    if end_cut - start_cut < NUM_SLICES:
        mid_z = all_z
    else:
        mid_z = all_z[start_cut:end_cut]

    # 在中间区间均匀采4张
    idx_arr = np.linspace(0, len(mid_z)-1, NUM_SLICES, dtype=int)
    sample_ids = mid_z[idx_arr]
    return sample_ids


# ===================== 2. 批量构造4张切片输入（查询点限制在框内） =====================
def build_batch_input(img_vol, sample_ids, bbox, ori_H, ori_W):
    slices_list = []
    z_pos_list = []
    bbox_list = []
    empty_pts_list = []
    empty_lab_list = []

    # 解析并校验3D框坐标 [x0,y0,z0,x1,y1,z1]
    x0, y0, z0, x1, y1, z1 = bbox
    # 强制修正坐标顺序，确保 low < high
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)
    z_min, z_max = min(z0, z1), max(z0, z1)

    # 最终校验，确保范围有效
    x_min = max(0, x_min)
    x_max = min(ori_W, x_max)
    y_min = max(0, y_min)
    y_max = min(ori_H, y_max)

    if x_min >= x_max:
        raise ValueError(f"X轴框范围无效：x_min={x_min}, x_max={x_max}（原始bbox：{bbox}）")
    if y_min >= y_max:
        raise ValueError(f"Y轴框范围无效：y_min={y_min}, y_max={y_max}（原始bbox：{bbox}）")

    # 归一化框坐标
    bbox_norm = np.array([x_min / ori_W, y_min / ori_H, x_max / ori_W, y_max / ori_H], dtype=np.float32)

    for z in sample_ids:
        img_slice = img_vol[z]
        img_pad = _resize_to_1024(img_slice)
        img_pad = (img_pad - img_pad.min()) / np.clip(img_pad.max() - img_pad.min(), 1e-8, None)
        img_tensor = torch.from_numpy(img_pad).float().unsqueeze(0).repeat(3, 1, 1)
        slices_list.append(img_tensor)
        z_pos_list.append(float(z))
        bbox_list.append(torch.from_numpy(bbox_norm).float())
        empty_pts_list.append(torch.zeros(1, 2))
        empty_lab_list.append(torch.zeros(1))

    slices_tensor = torch.stack(slices_list, dim=0).to(DEVICE)
    z_tensor = torch.tensor(z_pos_list).float().to(DEVICE)
    bbox_tensor = torch.stack(bbox_list, dim=0).to(DEVICE)
    pts_tensor = (
        torch.stack(empty_pts_list, dim=0).to(DEVICE),
        torch.stack(empty_lab_list, dim=0).to(DEVICE)
    )

    # 全局查询点：仅在框范围内生成（核心修改，修复randint边界问题）
    D_total = img_vol.shape[0]
    query_pts = []
    # 确保采样数量不超过有效坐标总数，避免无限循环
    max_possible_pts = (x_max - x_min) * (y_max - y_min) * len(sample_ids)
    safe_num_query = min(NUM_QUERY, max_possible_pts) if max_possible_pts > 0 else 100

    for _ in range(safe_num_query):
        # 修复：randint的high是开区间，所以x取到x_max（原x1），y取到y_max（原y1）
        x = np.random.randint(x_min, x_max)  # [x_min, x_max) 确保x_min < x_max
        y = np.random.randint(y_min, y_max)  # [y_min, y_max) 确保y_min < y_max
        # z: 仅在有效切片范围内采样（进一步缩小范围）
        z = np.random.choice(sample_ids) if len(sample_ids) > 0 else np.random.randint(0, D_total)
        query_pts.append([x, y, z])

    # 兜底：如果采样点为空，生成默认点
    if not query_pts:
        query_pts = [[(x_min + x_max) // 2, (y_min + y_max) // 2, sample_ids[0]] if len(sample_ids) > 0 else 0]

    query_tensor = torch.tensor(query_pts).float().to(DEVICE)
    raw_query_np = np.array(query_pts)
    return slices_tensor, z_tensor, bbox_tensor, pts_tensor, query_tensor, raw_query_np


# ===================== 3. 批量推理（4张一次性输入） =====================
@torch.no_grad()
def batch_sdf_inference(model, slices_tensor, z_tensor, bbox_tensor, pts_tensor, query_tensor):
    pred_list = []
    # 不再拆分query，全程共用一套全局查询点
    for idx, sub_slice in enumerate(torch.split(slices_tensor, SLICE_BATCH_SIZE, dim=0)):
        z_sub = torch.split(z_tensor, SLICE_BATCH_SIZE, dim=0)[idx]
        box_sub = torch.split(bbox_tensor, SLICE_BATCH_SIZE, dim=0)[idx]
        p0_sub = torch.split(pts_tensor[0], SLICE_BATCH_SIZE, dim=0)[idx]
        p1_sub = torch.split(pts_tensor[1], SLICE_BATCH_SIZE, dim=0)[idx]
        # query_tensor全程不变，统一传入
        sdf_sub = model(
            slices=sub_slice,
            query_points=query_tensor,
            slice_z_positions=z_sub,
            points_per_slice=(p0_sub, p1_sub),
            boxes_per_slice=box_sub
        )
        pred_list.append(sdf_sub)
    all_sdf = torch.cat(pred_list, dim=0)
    return all_sdf.cpu()


# ===================== 4. SDF重建网格 =====================
def full_sdf_to_mesh(img_vol, slice_sdf_vals, query_coords, level=0.0):
    D, H, W = img_vol.shape
    z_grid, y_grid, x_grid = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    full_sdf = np.zeros((D, H, W), dtype=np.float32)

    sdf_data = slice_sdf_vals[0]
    pts_data = query_coords
    print("坐标点数量：", len(pts_data))
    print("SDF数值数量：", len(sdf_data))

    # 修复插值：cubic + 减小填充值
    full_sdf = griddata(
        points=pts_data,
        values=sdf_data,
        xi=(z_grid, y_grid, x_grid),
        method="linear",
        fill_value=2.0
    )
    full_sdf = np.nan_to_num(full_sdf, nan=2.0)

    s_min = full_sdf.min()
    s_max = full_sdf.max()
    print(f"完整SDF场范围 min={s_min:.2f}, max={s_max:.2f}")
    real_level = 0.0

    # 第一层兜底：区间完全不包含0，直接调整阈值
    if not (s_min < 0 < s_max):
        print("警告：全场不含0，自动取中点阈值")
        real_level = (s_min + s_max) / 2

    # 第二层兜底：调整后依然全场同侧，直接返回空
    if s_min >= real_level:
        print("【致命】全场SDF均大于等值面，无器官表面")
        empty_mesh = trimesh.Trimesh()
        return empty_mesh, full_sdf
    if s_max <= real_level:
        print("【致命】全场SDF均小于等值面，无外表面")
        empty_mesh = trimesh.Trimesh()
        return empty_mesh, full_sdf

    # 打印诊断信息
    print("====SDF诊断信息====")
    print(f"SDF全场最小值 s_min={s_min:.3f}")
    print(f"SDF全场最大值 s_max={s_max:.3f}")
    print(f"当前等值面 real_level={real_level:.3f}")
    cross_mask = (full_sdf < real_level) & (np.roll(full_sdf, 1, axis=0) > real_level) \
              | (full_sdf > real_level) & (np.roll(full_sdf, 1, axis=0) < real_level) \
              | (full_sdf < real_level) & (np.roll(full_sdf, 1, axis=1) > real_level) \
              | (full_sdf > real_level) & (np.roll(full_sdf, 1, axis=1) < real_level) \
              | (full_sdf < real_level) & (np.roll(full_sdf, 1, axis=2) > real_level) \
              | (full_sdf > real_level) & (np.roll(full_sdf, 1, axis=2) < real_level)
    cross_count = np.sum(cross_mask)
    print(f"穿过等值面的体素数量：{cross_count}")
    print("===================")

    try:
        verts, faces, _, _ = marching_cubes(full_sdf, level=real_level)
    except ValueError:
        print("等值面不存在，重新调整阈值")
        real_level = (s_min + s_max) / 2
        verts, faces, _, _ = marching_cubes(full_sdf, level=real_level)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    #mesh.smooth()
    return mesh, full_sdf


# ===================== 主流程 =====================
if __name__ == "__main__":
    print("加载 SDF-SAM 模型...")
    model = build_sam_sdf(pretrained_path="medsam_vit_b.pth")
    model.load_state_dict(torch.load(MODEL_WEIGHT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 读取3D NIfTI
    print("加载3D医学体数据...")
    nii_img = nib.load(NII_PATH)
    img_data = nii_img.get_fdata()
    img_vol = img_data.transpose((2, 0, 1))  # (D, H, W)
    D, H, W = img_vol.shape
    print(f"3D体尺寸: D={D}, H={H}, W={W}")

    # 仅在框范围内采样4张有效切片（修改函数调用）
    print("框内均匀采样4张切片...")
    sample_slice_ids = get_valid_slices_in_box(img_vol, GLOBAL_BOX, H, W)
    print(f"采样切片索引: {sample_slice_ids.tolist()}")

    # 构造4张切片批量输入
    slices_tensor, z_tensor, bbox_tensor, pts_tensor, query_tensor, raw_query_np = build_batch_input(
        img_vol, sample_slice_ids, GLOBAL_BOX, H, W
    )
    print(f"输入切片批量维度: {slices_tensor.shape}")  # torch.Size([4, 3, 1024, 1024])
    print(f"查询点维度: {query_tensor.shape}")  # [NUM_QUERY, 3]

    # 批量推理
    print("批量推理4张切片SDF...")
    sdf_vals = batch_sdf_inference(model, slices_tensor, z_tensor, bbox_tensor, pts_tensor, query_tensor)
    print(f"SDF预测输出维度: {sdf_vals.shape}")

    # 生成3D网格
    print("生成3D表面网格...")
    mesh, _ = full_sdf_to_mesh(img_vol, sdf_vals, raw_query_np)

    # 保存&可视化
    mesh.export("./4slice_organ_mesh.obj")
    print("生成网格已保存到 4slice_organ_mesh.obj")
    print(f"顶点={len(mesh.vertices)}, 面片={len(mesh.faces)}")
    mesh.show()