# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import nibabel as nib
from skimage.measure import marching_cubes
import trimesh
import tempfile
from segment_anything import build_sam_sdf
import torch.nn.functional as F

from scipy.interpolate import griddata
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# ===================== 全局超参（修改切片数量为4） =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SLICES = 4  # 改为4张切片输入
SLICE_BATCH_SIZE = 12  # 批次大小等于总切片，一次推理完成
NUM_QUERY = 100000  # structured bbox grid budget (not random samples)
QUERY_CHUNK_SIZE = 20000
TARGET_ORGAN_ID = 9
# 路径配置
MODEL_WEIGHT = "result/best_sdf_sam_slice_24_plane.pth"
NII_PATH = "data/FLARE22Train/images/FLARE22_Tr_0002_0000.nii.gz"
# # 从3D Slicer导出的像素框 [x0, y0, z0, x1, y1, z1]
# GLOBAL_BOX = [70, 109, 21, 218, 215, 75]


# ===================== 工具函数 =====================
def get_single_organ_box(lab_vol, target_id, padding=2):
    """仅根据单个器官生成包围盒，忽略其余器官"""
    single_mask = (lab_vol == target_id)
    z_coords, y_coords, x_coords = np.where(single_mask > 0)
    if len(z_coords) == 0:
        D,H,W = lab_vol.shape
        return [0, 0, 0, W-1, H-1, D-1]

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
    return [xmin, ymin, zmin, xmax, ymax, zmax]

# def get_auto_global_box(lab_vol, padding=2):
#     """
#     根据真值掩码自动生成3D器官包围盒
#     :param lab_vol: (D, H, W) 维度label掩码，>0为器官
#     :param padding: 框向外扩充像素，避免截断器官边缘
#     :return: [xmin, ymin, zmin, xmax, ymax, zmax] 像素包围盒
#     """
#     # 获取所有器官体素坐标 (z,y,x)
#     z_coords, y_coords, x_coords = np.where(lab_vol > 0)
#     if len(z_coords) == 0:
#         # 无器官时返回整个图像作为兜底框
#         D, H, W = lab_vol.shape
#         return [0, 0, 0, W - 1, H - 1, D - 1]
#
#     # 原始极值
#     xmin_raw = x_coords.min()
#     xmax_raw = x_coords.max()
#     ymin_raw = y_coords.min()
#     ymax_raw = y_coords.max()
#     zmin_raw = z_coords.min()
#     zmax_raw = z_coords.max()
#
#     D, H, W = lab_vol.shape
#     # 向外padding，并限制不越界
#     xmin = max(0, xmin_raw - padding)
#     xmax = min(W - 1, xmax_raw + padding)
#     ymin = max(0, ymin_raw - padding)
#     ymax = min(H - 1, ymax_raw + padding)
#     zmin = max(0, zmin_raw - padding)
#     zmax = min(D - 1, zmax_raw + padding)
#
#     # 输出顺序 [x0,y0,z0,x1,y1,z1] 和你原GLOBAL_BOX格式完全统一
#     return [xmin, ymin, zmin, xmax, ymax, zmax]


def sample_axis_slices(vol, full_bbox, axis, num_slices):
    """
    vol: (D,H,W) 3D体
    full_bbox: [x0, y0, z0, x1, y1, z1] 全局3D包围盒
    axis: 0=XY(Z轴切片), 1=XZ(Y轴切片), 2=YZ(X轴切片)
    return slice_tensor, rel_tensor, plane_box_tensor
    """
    D, H, W = vol.shape
    # 拆解全局包围盒，所有坐标来源统一，无陌生变量
    x0, y0, z0, x1, y1, z1 = full_bbox
    # 各轴有效采样区间
    if axis == 0:
        # XY切片：沿Z轴取片，Z范围 [z0, z1]
        a_min, a_max = z0, z1
        # XY平面归一框，和dataset公式完全一致
        plane_box = np.array([x0 / W, y0 / H, x1 / W, y1 / H], dtype=np.float32)
    elif axis == 1:
        # XZ切片：沿Y轴取片，Y范围 [y0, y1]
        a_min, a_max = y0, y1
        # XZ平面归一框
        plane_box = np.array([x0 / W, z0 / D, x1 / W, z1 / D], dtype=np.float32)
    else:
        # YZ切片：沿X轴取片，X范围 [x0, x1]
        a_min, a_max = x0, x1
        # YZ平面归一框
        plane_box = np.array([y0 / H, z0 / D, y1 / H, z1 / D], dtype=np.float32)

    # Cover the same closed bbox interval as reconstruction queries. The first
    # and last sparse slices explicitly bracket every queried coordinate.
    sample_ids = np.rint(np.linspace(a_min, a_max, num_slices)).astype(int)

    slices_list = []
    rel_list = []
    box_list = []
    for a in sample_ids:
        # 取出对应2D切片
        if axis == 0:
            slc = vol[a, :, :]
        elif axis == 1:
            slc = vol[:, a, :]
        else:
            slc = vol[:, :, a]
        # Keep the native voxel grid; only intensity normalization is applied.
        native = np.asarray(slc, dtype=np.float32)
        native = (native - native.min()) / (native.max() - native.min() + 1e-8)
        slice_tensor = torch.from_numpy(native).unsqueeze(0).repeat(3, 1, 1)
        slices_list.append(slice_tensor)
        # Same voxel-coordinate frame as query_points.
        rel_list.append(float(a))
        box_list.append(torch.from_numpy(plane_box).float())

    slice_tensor = torch.stack(slices_list).to(DEVICE)
    rel_tensor = torch.tensor(rel_list).float().to(DEVICE)
    box_tensor = torch.stack(box_list).to(DEVICE)
    return slice_tensor, rel_tensor, box_tensor
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


def build_batch_input(img_vol, bbox, num_slices=8):
    """
    bbox: [x0, y0, z0, x1, y1, z1] 全局3D包围盒
    返回三平面全套输入，完全匹配model forward参数
    """
    D, H, W = img_vol.shape
    x0, y0, z0, x1, y1, z1 = bbox
    # 分别生成XY / XZ / YZ 三组切片、rel、平面框
    xy_slices, xy_rel, xy_box = sample_axis_slices(img_vol, bbox, axis=0, num_slices=num_slices)
    xz_slices, xz_rel, xz_box = sample_axis_slices(img_vol, bbox, axis=1, num_slices=num_slices)
    yz_slices, yz_rel, yz_box = sample_axis_slices(img_vol, bbox, axis=2, num_slices=num_slices)

    # Build a regular, aspect-ratio-preserving grid. Random points followed by
    # integer rounding left duplicate samples and holes in the interpolated SDF.
    extents = np.array([x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1], dtype=np.float64)
    scale = min(1.0, (NUM_QUERY / np.prod(extents)) ** (1.0 / 3.0))
    counts = np.maximum(2, np.ceil(extents * scale).astype(int))
    xs = np.linspace(x0, x1, counts[0], dtype=np.float32)
    ys = np.linspace(y0, y1, counts[1], dtype=np.float32)
    zs = np.linspace(z0, z1, counts[2], dtype=np.float32)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    query_pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    raw_query_np = query_pts.copy()
    query_tensor = torch.tensor(query_pts).float().to(DEVICE)

    # 空点占位，推理传None等价
    empty_pts = torch.zeros(num_slices, 1, 2).to(DEVICE)
    empty_lab = torch.zeros(num_slices, 1).to(DEVICE)
    pts_tensor = (empty_pts, empty_lab)

    return (
        xy_slices, xz_slices, yz_slices,
        xy_rel, xz_rel, yz_rel,
        xy_box, xz_box, yz_box,
        pts_tensor, query_tensor, raw_query_np
    )

# ===================== 3. 批量推理（4张一次性输入） =====================
@torch.no_grad()
def batch_sdf_inference(
    model,
    xy_slices, xz_slices, yz_slices,
    xy_rel, xz_rel, yz_rel,
    xy_boxes, xz_boxes, yz_boxes,
    query_tensor,
    SLICE_BATCH_SIZE=8,
    query_chunk_size=QUERY_CHUNK_SIZE,
):
    model.eval()
    pred_list = []
    # All sparse slices must participate in every prediction. Chunk queries,
    # never slices; concatenating predictions from different slice subsets is
    # not a valid SDF.
    xy_sub, xz_sub, yz_sub = (x.unsqueeze(0) for x in (xy_slices, xz_slices, yz_slices))
    xy_rel_sub, xz_rel_sub, yz_rel_sub = (x.unsqueeze(0) for x in (xy_rel, xz_rel, yz_rel))
    xy_box_sub, xz_box_sub, yz_box_sub = (x.unsqueeze(0) for x in (xy_boxes, xz_boxes, yz_boxes))
    for start in range(0, query_tensor.shape[0], query_chunk_size):
        q_batch = query_tensor[start:start + query_chunk_size].unsqueeze(0)
        sdf_sub = model(
            xy_slices=xy_sub,
            xz_slices=xz_sub,
            yz_slices=yz_sub,
            xy_rel=xy_rel_sub,
            xz_rel=xz_rel_sub,
            yz_rel=yz_rel_sub,
            xy_boxes=xy_box_sub,
            xz_boxes=xz_box_sub,
            yz_boxes=yz_box_sub,
            query_points=q_batch,
            points_per_slice=None
        )
        pred_list.append(sdf_sub.squeeze(0).cpu())
    full_pred = torch.cat(pred_list, dim=0)
    return full_pred
# ===================== 4. SDF重建网格 =====================
def full_sdf_to_mesh(img_vol, lab_vol, slice_sdf_vals, query_coords, bbox, sample_slice_ids, nii_affine, level=0.0):
    x0, y0, z0, x1, y1 ,z1= bbox
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)
    z_min, z_max = min(z0, z1), max(z0, z1)
    D, H, W = img_vol.shape

    # 取出模型预测SDF
    if torch.is_tensor(slice_sdf_vals):
        sdf_data = slice_sdf_vals.detach().cpu().numpy()
    else:
        sdf_data = slice_sdf_vals
    pts_data = np.asarray(query_coords)

    print("坐标点数量：", len(pts_data))
    print("SDF数值数量：", len(sdf_data))
    print(f"原始归一SDF min={sdf_data.min():.4f}, max={sdf_data.max():.4f}")

    # Interpolate only inside the prompted bbox and retain floating query
    # coordinates. Outside is positive TSDF background. This avoids both the
    # huge whole-volume griddata call and artificial fill-value cavities.
    # query_coords本身就是规则3D网格，不要再使用scipy.griddata
    xs = np.unique(pts_data[:, 0])
    ys = np.unique(pts_data[:, 1])
    zs = np.unique(pts_data[:, 2])
    nx, ny, nz = len(xs), len(ys), len(zs)
    print(f"规则查询网格尺寸: nz={nz}, ny={ny}, nx={nx}, 总点数={nz * ny * nx}")
    if nz * ny * nx != len(sdf_data):
        raise ValueError(f"查询点不是完整规则网格: {nz}*{ny}*{nx}={nz * ny * nx}, 实际SDF={len(sdf_data)}")

    # build_batch_input中的flatten顺序就是[z,y,x]，所以直接reshape
    coarse_sdf = sdf_data.reshape(nz, ny, nx).astype(np.float32)
    print(f"粗SDF网格: {coarse_sdf.shape}, min={coarse_sdf.min():.4f}, max={coarse_sdf.max():.4f}")

    # 目标bbox体素网格
    zi = np.arange(int(np.floor(z_min)), int(np.ceil(z_max)) + 1)
    yi = np.arange(int(np.floor(y_min)), int(np.ceil(y_max)) + 1)
    xi = np.arange(int(np.floor(x_min)), int(np.ceil(x_max)) + 1)

    # 用PyTorch三线性插值：规则网格→规则网格，速度远快于griddata
    coarse_tensor = torch.from_numpy(coarse_sdf).unsqueeze(0).unsqueeze(0)
    local_sdf = F.interpolate(
        coarse_tensor,
        size=(len(zi), len(yi), len(xi)),
        mode="trilinear",
        align_corners=True
    ).squeeze(0).squeeze(0).numpy()

    full_sdf = np.ones((D, H, W), dtype=np.float32)
    full_sdf[zi[0]:zi[-1] + 1, yi[0]:yi[-1] + 1, xi[0]:xi[-1] + 1] = local_sdf
    print(f"三线性插值完成: {local_sdf.shape}, min={local_sdf.min():.4f}, max={local_sdf.max():.4f}")


    # # 3、值域矫正：仅框内做均值归零+拉伸
    # valid_area = full_sdf[full_sdf < 9.0]
    # mean_valid = np.mean(valid_area)
    # full_sdf = full_sdf - mean_valid
    # stretch_scale = 1.0
    # full_sdf[full_sdf < 9.0] *= stretch_scale

    s_min = local_sdf.min()
    s_max = local_sdf.max()
    print(f"拉伸矫正后框内SDF min={s_min:.3f}, max={s_max:.3f}")

    # 统计正负采样点
    pos_count = np.sum(sdf_data > 0)
    neg_count = np.sum(sdf_data < 0)
    zero_count = np.sum(np.isclose(sdf_data, 0, atol=1e-4))
    print(f"【SDF数值统计】内部负点:{neg_count} | 外部正点:{pos_count} | 近表面点:{zero_count}")

    real_level = 0.0
    if not (s_min < 0 < s_max):
        raise RuntimeError(
            f"Predicted TSDF does not cross zero (min={s_min:.4f}, max={s_max:.4f}); "
            "the checkpoint has not learned a valid signed field yet."
        )

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
        N = pix_verts.shape[0]
        # 拆分：marching_cubes输出顺序 [z,y,x]
        z_corner = pix_verts[:, 0]
        y_corner = pix_verts[:, 1]
        x_corner = pix_verts[:, 2]

        # 角点 -> 体素中心偏移 +0.5
        x_center = x_corner + 0.5
        y_center = y_corner + 0.5
        z_center = z_corner + 0.5

        # 构造齐次坐标 [X,Y,Z, 1] 4行N列
        pts_ijk = np.stack([x_center, y_center, z_center, np.ones(N)], axis=0)
        # IJK像素中心 → LPS毫米坐标
        lps_all = affine @ pts_ijk
        l_x, l_y, l_z = lps_all[:3]

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
    lab_restore = lab_vol.transpose((1, 2, 0))
    label_nii = nib.Nifti1Image(lab_restore, nii_affine)
    nib.save(label_nii, "./gt_organ_label.nii.gz")
    print("【对比真值】gt_organ_label.nii.gz 已保存")

    # 由SDF生成器官二值mask：SDF<0 = 器官内部
    organ_mask = (full_sdf < 0).astype(np.float32)

    # 关键：D,H,W 转回原图原始 (H,W) 维度匹配affine
    mask_save = organ_mask.transpose((1, 2, 0))
    mask_nii = nib.Nifti1Image(mask_save, nii_affine)
    nib.save(mask_nii, "./pred_organ_mask.nii.gz")
    print("预测器官掩码已保存 pred_organ_mask.nii.gz，可直接在Slicer和CT叠加查看")
    organ_mask = (full_sdf < 0).astype(np.uint8)  # 关键：转uint8整数，不是float
    # D,H,W → 还原原始CT H,W,D维度
    mask_save = organ_mask.transpose((1, 2, 0))
    # 文件名带上label，让Slicer自动识别为标签图
    mask_nii = nib.Nifti1Image(mask_save, nii_affine)
    nib.save(mask_nii, "./pred_label.nii.gz")
    print("输出pred_label.nii.gz，和gt_organ_label完全一致格式")
    # sdf_restore = full_sdf.transpose((1, 2, 0))
    # sdf_nii = nib.Nifti1Image(sdf_restore, nii_affine)
    # nib.save(sdf_nii, "./pred_sdf_volume.nii.gz")
    # print("稠密SDF体文件 pred_sdf_volume.nii.gz 已生成，Slicer加载CT直接对齐无偏移")

    # ========= 合并对比场景（像素坐标，本地查看用） =========
    scene = trimesh.Scene()
    scene.add_geometry(gt_mesh, "gt_organ_green")
    scene.add_geometry(pred_mesh, "pred_organ_gray")
    scene.export("./compare_pred_gt.obj")
    print("【本地查看】对比网格 compare_pred_gt.obj")

    return pred_mesh, full_sdf

if __name__ == "__main__":
    print("加载 SDF-SAM 模型...")

    model = build_sam_sdf(pretrained_path="result/best_sdf_sam_slice_24_plane.pth")
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
    GLOBAL_BOX = get_single_organ_box(lab_vol, TARGET_ORGAN_ID, padding=10)
    # GLOBAL_BOX =[70, 109, 21, 218, 215, 72]
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
    # 原调用删掉，替换
    xy_slices, xz_slices, yz_slices, \
        xy_rel, xz_rel, yz_rel, \
        xy_box, xz_box, yz_box, \
        pts_tensor, query_tensor, raw_query_np = build_batch_input(
        img_vol, GLOBAL_BOX, num_slices=NUM_SLICES
    )
    # print(f"输入切片批量维度: {slices_tensor.shape}")  # torch.Size([4, 3, 1024, 1024])
    print(f"查询点维度: {query_tensor.shape}")  # [NUM_QUERY, 3]

    # 批量推理
    print("批量推理24张切片SDF...")
    sdf_vals = batch_sdf_inference(
        model=model,
        xy_slices=xy_slices,
        xz_slices=xz_slices,
        yz_slices=yz_slices,
        xy_rel=xy_rel,
        xz_rel=xz_rel,
        yz_rel=yz_rel,
        xy_boxes=xy_box,
        xz_boxes=xz_box,
        yz_boxes=yz_box,
        query_tensor=query_tensor,
        SLICE_BATCH_SIZE=SLICE_BATCH_SIZE
    )
    print(f"SDF预测输出维度: {sdf_vals.shape}")

    # 生成3D网格
    print("生成3D表面网格...")
    single_organ_lab = (lab_vol == TARGET_ORGAN_ID).astype(np.float32)
    mesh, _ = full_sdf_to_mesh(
        img_vol,
        single_organ_lab,
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
