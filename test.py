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
NUM_SLICES = 8  # 改为4张切片输入
SLICE_BATCH_SIZE = 8  # 批次大小等于总切片，一次推理完成
NUM_QUERY = 50000 # 每张切片的查询点数量
# 路径配置
MODEL_WEIGHT = "best_sdf_sam_slice_cross_transformer_sin.pth"
NII_PATH = "data/FLARE22Train/images/FLARE22_Tr_0002_0000.nii.gz"
# # 从3D Slicer导出的像素框 [x0, y0, z0, x1, y1, z1]
# GLOBAL_BOX = [70, 109, 21, 218, 215, 75]


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


def get_auto_global_box(lab_vol, padding=2):
    """
    根据真值掩码自动生成3D器官包围盒
    :param lab_vol: (D, H, W) 维度label掩码，>0为器官
    :param padding: 框向外扩充像素，避免截断器官边缘
    :return: [xmin, ymin, zmin, xmax, ymax, zmax] 像素包围盒
    """
    # 获取所有器官体素坐标 (z,y,x)
    z_coords, y_coords, x_coords = np.where(lab_vol > 0)
    if len(z_coords) == 0:
        # 无器官时返回整个图像作为兜底框
        D, H, W = lab_vol.shape
        return [0, 0, 0, W - 1, H - 1, D - 1]

    # 原始极值
    xmin_raw = x_coords.min()
    xmax_raw = x_coords.max()
    ymin_raw = y_coords.min()
    ymax_raw = y_coords.max()
    zmin_raw = z_coords.min()
    zmax_raw = z_coords.max()

    D, H, W = lab_vol.shape
    # 向外padding，并限制不越界
    xmin = max(0, xmin_raw - padding)
    xmax = min(W - 1, xmax_raw + padding)
    ymin = max(0, ymin_raw - padding)
    ymax = min(H - 1, ymax_raw + padding)
    zmin = max(0, zmin_raw - padding)
    zmax = min(D - 1, zmax_raw + padding)

    # 输出顺序 [x0,y0,z0,x1,y1,z1] 和你原GLOBAL_BOX格式完全统一
    return [xmin, ymin, zmin, xmax, ymax, zmax]

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

    query_pts = []
    for _ in range(NUM_QUERY):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)
        query_pts.append([x, y, z])
    query_pts = np.array(query_pts, dtype=np.float32)

    # 兜底：如果采样点为空，生成框中心单点
    if len(query_pts) == 0:
        query_pts = np.array([
            [(x_min + x_max) // 2, (y_min + y_max) // 2, (z_min + z_max) // 2]
        ], dtype=np.float32)
    query_tensor = torch.tensor(query_pts).float().to(DEVICE)
    raw_query_np = np.array(query_pts)
    return slices_tensor, z_tensor, bbox_tensor, pts_tensor, query_tensor, raw_query_np


# ===================== 3. 批量推理（4张一次性输入） =====================
@torch.no_grad()
def batch_sdf_inference(model, slices_tensor, z_tensor, bbox_tensor, pts_tensor, query_tensor):
    """
    适配 SDF-SAM 原始模型版本的稳定推理脚本
    """

    model.eval()
    pred_list = []

    B, S = slices_tensor.shape[0], slices_tensor.shape[0]  # 4 slices

    # ==============================
    # 1. 保证输入维度安全
    # ==============================
    # slices: [S, 3, H, W]
    # z_tensor: [S]
    # bbox_tensor: [S, 4]
    # pts_tensor: (points, labels)

    points, labels = pts_tensor  # points:[S,1,2] labels:[S,1]

    # ==============================
    # 2. 扩展为 SAM 期望格式
    # ==============================
    # SAM期望: [B, N, 2]
    points = points.unsqueeze(0)   # [1, S, 1, 2]
    labels = labels.unsqueeze(0)   # [1, S, 1]

    bbox_tensor = bbox_tensor.unsqueeze(0)  # [1, S, 4]
    z_tensor = z_tensor.unsqueeze(0)        # [1, S]

    # query保持不变: [N, 3]
    query_tensor = query_tensor

    # ==============================
    # 3. forward batch inference
    # ==============================
    for idx, sub_slice in enumerate(torch.split(slices_tensor, SLICE_BATCH_SIZE, dim=0)):

        z_sub = z_tensor[:, idx:idx+len(sub_slice)]           # [1, S_sub]
        box_sub = bbox_tensor[:, idx:idx+len(sub_slice)]      # [1, S_sub, 4]

        # ⭐关键修复：不要破坏 [B,S,N,2]
        p_sub = points[:, idx:idx+len(sub_slice), :, :]       # [1, S_sub, 1, 2]
        l_sub = labels[:, idx:idx+len(sub_slice), :]          # [1, S_sub, 1]

        print("coords:", p_sub.shape)
        print("labels:", l_sub.shape)
        query_tensor = query_tensor.unsqueeze(0)
        sdf_sub = model(
            slices=sub_slice,
            query_points=query_tensor,
            slice_z_positions=z_sub.squeeze(0),
            points_per_slice=(p_sub, l_sub),
            boxes_per_slice=box_sub
        )

        pred_list.append(sdf_sub)

    return torch.cat(pred_list, dim=0)
# ===================== 4. SDF重建网格 =====================
def full_sdf_to_mesh(img_vol, lab_vol, slice_sdf_vals, query_coords, bbox, sample_slice_ids, nii_affine, level=0.0):
    x0, y0, z0, x1, y1 ,z1= bbox
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)
    z_min, z_max = min(z0, z1), max(z0, z1)
    D, H, W = img_vol.shape

    # 取出模型预测SDF
    if torch.is_tensor(slice_sdf_vals):
        sdf_data = slice_sdf_vals[0].detach().cpu().numpy()
    else:
        sdf_data = slice_sdf_vals[0]
    pts_data = np.asarray(query_coords)

    print("坐标点数量：", len(pts_data))
    print("SDF数值数量：", len(sdf_data))
    print(f"原始归一SDF min={sdf_data.min():.4f}, max={sdf_data.max():.4f}")

    # 1、初始化全场：框外填充超大正数10.0（代表远背景）
    full_sdf = np.full((D, H, W), fill_value=10.0, dtype=np.float32)
    valid_mask = np.zeros((D, H, W), dtype=bool)
    for i, p in enumerate(pts_data):
        # p = [x, y, z]，volume索引是 [z, y, x]
        px, py, pz = p.astype(int)
        z = pz
        y = py
        x = px
        if 0 <= z < D and 0 <= y < H and 0 <= x < W:
            full_sdf[z, y, x] = sdf_data[i]
            valid_mask[z, y, x] = True


    # 2、线性插值补全框内连续SDF场
    valid_pts = np.argwhere(valid_mask)
    valid_vals = full_sdf[valid_mask]
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    grid_pts = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
    full_sdf = griddata(valid_pts, valid_vals, grid_pts, method="linear", fill_value=10.0)
    full_sdf = full_sdf.reshape(D, H, W)



    # 3、值域矫正：仅框内做均值归零+拉伸
    valid_area = full_sdf[full_sdf < 9.0]
    mean_valid = np.mean(valid_area)
    full_sdf = full_sdf - mean_valid
    stretch_scale = 1.0
    full_sdf[full_sdf < 9.0] *= stretch_scale

    s_min = full_sdf[full_sdf < 9.0].min()
    s_max = full_sdf[full_sdf < 9.0].max()
    print(f"拉伸矫正后框内SDF min={s_min:.3f}, max={s_max:.3f}")

    # 统计正负采样点
    pos_count = np.sum(sdf_data > 0)
    neg_count = np.sum(sdf_data < 0)
    zero_count = np.sum(np.isclose(sdf_data, 0, atol=1e-4))
    print(f"【SDF数值统计】内部负点:{neg_count} | 外部正点:{pos_count} | 近表面点:{zero_count}")

    real_level = 0.0
    if not (s_min < 0 < s_max):
        real_level = np.percentile(full_sdf[full_sdf < 9.0], 48)

    # 等值面统计
    # 统计有效等值（只统计框内）
    cross_mask = ((full_sdf < real_level) & (np.roll(full_sdf, shift=1, axis=0) > real_level)) \
                 | ((full_sdf > real_level) & (np.roll(full_sdf, shift=1, axis=0) < real_level)) \
                 | ((full_sdf < real_level) & (np.roll(full_sdf, shift=1, axis=1) > real_level)) \
                 | ((full_sdf > real_level) & (np.roll(full_sdf, shift=1, axis=1) < real_level)) \
                 | ((full_sdf < real_level) & (np.roll(full_sdf, shift=2, axis=2) > real_level)) \
                 | ((full_sdf > real_level) & (np.roll(full_sdf, shift=2, axis=2) < real_level))
    cross_count = np.sum(cross_mask & (full_sdf < 9.0))
    print("====SDF诊断信息====")
    print(f"框内最小值 s_min={s_min:.3f}")
    print(f"框内最大值 s_max={s_max:.3f}")
    print(f"提取等值面 level={real_level:.3f}")
    print(f"器官有效交叉体素：{cross_count}")
    print("===================")

    # try:
    #     verts_pred, faces_pred, _, _ = marching_cubes(full_sdf, level=real_level)
    # except ValueError:
    #     real_level = np.percentile(full_sdf[full_sdf < 9.0], 50)
    #     verts_pred, faces_pred, _, _ = marching_cubes(full_sdf, level=real_level)
    # mesh = trimesh.Trimesh(vertices=verts_pred, faces=faces_pred)
    # mesh.export("./4slice_organ_mesh.obj")
    # mesh.show()
    try:
        verts_pred_pix, faces_pred, _, _ = marching_cubes(full_sdf, level=real_level)
    except ValueError:
        real_level = np.percentile(full_sdf[full_sdf < 9.0], 50)
        verts_pred_pix, faces_pred, _, _ = marching_cubes(full_sdf, level=real_level)
        # 真值像素网格
    mask_bin = (lab_vol > 0).astype(np.float32)
    mask_bin = (lab_vol > 0).astype(np.float32)
    vert_gt_pix, face_gt, _, _ = marching_cubes(mask_bin, level=0.1)

    def pix2world(pix_verts, affine):
        """
        pix_verts: N,3 [z, y, x] marching_cubes 体素角点整数坐标
        affine: nibabel 4×4 LPS仿射矩阵
        return N,3 RAS世界坐标（适配3D Slicer，修正体素中心偏移）
        """
        z_corner = pix_verts[:, 0]
        y_corner = pix_verts[:, 1]
        x_corner = pix_verts[:, 2]

        # 关键修复：marching_cubes输出是体素角，NIfTI坐标基于体素中心，+0.5偏移
        x_center = x_corner + 0.5
        y_center = y_corner + 0.5
        z_center = z_corner + 0.5

        # 构造齐次 IJK [X,Y,Z,1]
        N = len(x_center)
        pts_ijk = np.stack([x_center, y_center, z_center, np.ones(N)], axis=0)  # (4, N)

        # IJK像素中心 → LPS物理坐标
        lps_world = affine @ pts_ijk
        l_x, l_y, l_z = lps_world[:3]

        # LPS -> RAS标准转换：R=-L, A=-P, S=Z（自适应Z轴正负）
        ras_x = -l_x
        ras_y = -l_y
        # 判断Z轴方向，负数则翻转
        if affine[2, 2] < 0:
            ras_z = -l_z
        else:
            ras_z = l_z

        return np.stack([ras_x, ras_y, ras_z], axis=1)

    verts_pred_world = pix2world(verts_pred_pix, nii_affine)
    vert_gt_world = pix2world(vert_gt_pix, nii_affine)

    # 预测网格（世界坐标，灰色）
    pred_mesh = trimesh.Trimesh(vertices=verts_pred_world, faces=faces_pred)

    pred_mesh.export("./pred_organ_world.obj")
    print("【Slicer专用】预测器官世界坐标网格 pred_organ_world.obj")

    # 真值网格（世界坐标，绿色半透明）
    gt_mesh = trimesh.Trimesh(vertices=vert_gt_world, faces=face_gt)
    gt_mesh.visual.vertex_colors = np.full((len(gt_mesh.vertices), 3), [30, 220, 80], dtype=np.uint8)
    gt_mesh.export("./gt_organ_world.obj")
    gt_mesh.show()
    print("【Slicer专用】真值器官世界坐标网格 gt_organ_world.obj")

    # 导出真值label NII（同原始CT仿射）
    label_nii = nib.Nifti1Image(lab_vol, nii_affine)
    nib.save(label_nii, "./gt_organ_label.nii.gz")
    print("【对比真值】gt_organ_label.nii.gz 已保存")

    # ========= 合并对比场景（像素坐标，本地查看用） =========
    scene = trimesh.Scene()
    scene.add_geometry(gt_mesh, "gt_organ_green")
    scene.add_geometry(pred_mesh, "pred_organ_gray")
    scene.export("./compare_pred_gt.obj")
    print("【本地查看】对比网格 compare_pred_gt.obj")

    return pred_mesh, full_sdf





# ===================== 主流程 =====================

if __name__ == "__main__":
    print("加载 SDF-SAM 模型...")
    model = build_sam_sdf(pretrained_path="best_sdf_sam_slice_cross_transformer_sin.pth")
    model.load_state_dict(torch.load(MODEL_WEIGHT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 读取3D NIfTI
    print("加载3D医学体数据...")
    nii_img = nib.load(NII_PATH)
    img_data = nii_img.get_fdata()
    nii_affine = nii_img.affine  # 关键：空间转换矩阵，用于网格对齐
    img_vol = img_data.transpose((2, 0, 1))  # (D, H, W)
    D, H, W = img_vol.shape
    print(f"3D体尺寸: D={D}, H={H}, W={W}")
    LABEL_PATH = "data/FLARE22Train/labels/FLARE22_Tr_0002.nii.gz"
    label_nii = nib.load(LABEL_PATH)
    lab_data = label_nii.get_fdata()
    lab_vol = lab_data.transpose((2, 0, 1))
    print("Affine Z缩放系数：", nii_affine[2, 2])
    GLOBAL_BOX = get_auto_global_box(lab_vol, padding=10)
    print("自动基于真值生成Global Box：", GLOBAL_BOX)
    # x0, y0, z0, x1, y1, z1 = GLOBAL_BOX
    # x_min, x_max = min(x0, x1), max(x0, x1)
    # y_min, y_max = min(y0, y1), max(y0, y1)
    # z_min, z_max = min(z0, z1), max(z0, z1)
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
    mesh, _ = full_sdf_to_mesh(
        img_vol,
        lab_vol,
        sdf_vals,
        raw_query_np,
        GLOBAL_BOX,
        sample_slice_ids,
        nii_affine
    )

    # 保存&可视化
    # mesh.export("./4slice_organ_mesh.obj")
    # print("生成网格已保存到 4slice_organ_mesh.obj")
    print(f"顶点={len(mesh.vertices)}, 面片={len(mesh.faces)}")
    mesh.show()
