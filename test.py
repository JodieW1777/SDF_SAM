import os
import numpy as np
import torch
import cv2
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes
import trimesh
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ===================== 全局配置（与训练代码严格对齐） =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024
NUM_QUERY = 512
NUM_SLICES = 8        # 固定提取8张切片（和训练一致）
SLICE_BATCH_SIZE = 4  # 切片分片大小（显存优化，和训练一致）
# 模型权重路径
MODEL_WEIGHT = "./best_sdf_sam.pth"

# ===================== 复用训练脚本基础函数 =====================
def mask2sdf(mask):
    inside = distance_transform_edt(mask == 1)
    outside = distance_transform_edt(mask == 0)
    sdf = outside - inside
    return sdf.astype(np.float32)

def _resize_to_1024(img_slice, mask_slice, target_size=1024):
    h, w = img_slice.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_slice, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_slice, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    img_pad = np.zeros((target_size, target_size), dtype=np.float32)
    mask_pad = np.zeros((target_size, target_size), dtype=np.float32)
    offset_h = (target_size - new_h) // 2
    offset_w = (target_size - new_w) // 2
    img_pad[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = img_resized
    mask_pad[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = mask_resized
    return img_pad, mask_pad

# ===================== 模型加载（缓存加速） =====================
@st.cache_resource
def load_model(weight_path):
    from segment_anything import build_sam_sdf
    model = build_sam_sdf(pretrained_path="medsam_vit_b.pth")
    ckpt = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    return model

# ===================== 核心1：根据框范围 筛选+均匀采样8张切片 =====================
def get_slices_in_bbox(img_vol, rel_bbox, H, W, num_slices=8):
    """
    在画框范围内筛选有效切片，并均匀采样指定数量切片
    :param img_vol: 3D体数据 (D, H, W)
    :param rel_bbox: 归一化框 [xmin, ymin, xmax, ymax]
    :param H,W: 单切片原始高宽
    :param num_slices: 需要提取的切片数
    :return: 采样切片索引列表、所有有效切片索引列表
    """
    D = img_vol.shape[0]
    xmin_r, ymin_r, xmax_r, ymax_r = rel_bbox
    # 转为像素坐标
    xmin = int(xmin_r * W)
    xmax = int(xmax_r * W)
    ymin = int(ymin_r * H)
    ymax = int(ymax_r * H)

    valid_slice_ids = []
    for z in range(D):
        # 截取当前切片的框内区域
        roi = img_vol[z, ymin:ymax, xmin:xmax]
        # 框内区域有内容则判定为有效切片
        if np.any(roi > 0):
            valid_slice_ids.append(z)

    # 均匀采样8张切片
    if len(valid_slice_ids) >= num_slices:
        idx_list = np.linspace(0, len(valid_slice_ids)-1, num_slices, dtype=int)
        sample_slice_ids = np.array(valid_slice_ids)[idx_list]
    else:
        # 容错：有效切片不足，全局均匀采样
        sample_slice_ids = np.linspace(0, D-1, num_slices, dtype=int)

    return sample_slice_ids, valid_slice_ids

# ===================== 核心2：批量8切片推理（沿用训练分片逻辑） =====================
def infer_8_slices_batch(model, img_vol, sample_slice_ids, rel_bbox, H, W):
    """
    批量输入8张切片，分片推理，返回每张切片预测的SDF值
    """
    # 1. 构造模型输入张量（8张切片）
    slices_list = []
    z_pos_list = []
    bbox_list = []
    empty_pts_list = []
    empty_lab_list = []

    xmin_r, ymin_r, xmax_r, ymax_r = rel_bbox
    bbox_pix = np.array([xmin_r*W, ymin_r*H, xmax_r*W, ymax_r*H], dtype=np.float32)

    for z in sample_slice_ids:
        img_slice = img_vol[z]
        mask_slice = np.zeros_like(img_slice)
        img_pad, _ = _resize_to_1024(img_slice, mask_slice, IMG_SIZE)
        img_pad = (img_pad - img_pad.min()) / (img_pad.max() + 1e-8)
        # 转为3通道 (3, 1024, 1024)
        img_tensor = torch.from_numpy(img_pad).float().unsqueeze(0).repeat(3, 1, 1)
        slices_list.append(img_tensor)
        z_pos_list.append(float(z))
        bbox_list.append(torch.from_numpy(bbox_pix).float())
        empty_pts_list.append(torch.zeros(1, 2))
        empty_lab_list.append(torch.zeros(1))

    # 拼接成批量输入 (8, 3, 1024, 1024)
    slices_tensor = torch.stack(slices_list, dim=0).to(DEVICE)
    z_pos_tensor = torch.tensor(z_pos_list).float().to(DEVICE)
    bbox_tensor = torch.stack(bbox_list, dim=0).to(DEVICE)
    pts_tensor = (torch.stack(empty_pts_list, dim=0).to(DEVICE),
                  torch.stack(empty_lab_list, dim=0).to(DEVICE))

    # 2. 构造全局查询点 (NUM_QUERY, 3)
    D_total = img_vol.shape[0]
    query_pts = []
    for _ in range(NUM_QUERY):
        z = np.random.randint(0, D_total)
        y = np.random.randint(0, H)
        x = np.random.randint(0, W)
        query_pts.append([x, y, z])
    query_tensor = torch.tensor(query_pts).float().to(DEVICE)

    # 3. 分片推理（和训练逻辑完全一致）
    with torch.no_grad():
        slices_split = torch.split(slices_tensor, SLICE_BATCH_SIZE, dim=0)
        query_split = torch.split(query_tensor, SLICE_BATCH_SIZE, dim=0)
        z_split = torch.split(z_pos_tensor, SLICE_BATCH_SIZE, dim=0)
        bbox_split = torch.split(bbox_tensor, SLICE_BATCH_SIZE, dim=0)
        pts0_split = torch.split(pts_tensor[0], SLICE_BATCH_SIZE, dim=0)
        pts1_split = torch.split(pts_tensor[1], SLICE_BATCH_SIZE, dim=0)

        pred_list = []
        for idx, slice_sub in enumerate(slices_split):
            start_idx = idx * SLICE_BATCH_SIZE
            end_idx = start_idx + len(slice_sub)
            pred_sub = model(
                slices=slice_sub,
                query_points=query_split[idx],
                slice_z_positions=z_split[idx],
                points_per_slice=(pts0_split[idx], pts1_split[idx]),
                boxes_per_slice=bbox_split[idx]
            )
            pred_list.append(pred_sub)

        # 拼接8张切片的预测结果
        all_pred = torch.cat(pred_list, dim=0)

    # 返回：切片索引 + 对应预测SDF值
    sdf_vals = all_pred.detach().cpu().numpy()
    return dict(zip(sample_slice_ids, sdf_vals))

# ===================== 核心3：补全全体积SDF场 =====================
def fill_3d_sdf(img_vol, slice_sdf_dict, default_sdf=0.0):
    """用8张切片预测值补全整个3D体SDF"""
    D, H, W = img_vol.shape
    full_sdf = np.full((D, H, W), default_sdf, dtype=np.float32)
    # 填充已预测的8张切片
    for z, val in slice_sdf_dict.items():
        full_sdf[int(z)] = val
    return full_sdf

# ===================== SDF转3D网格 =====================
def sdf_to_mesh(sdf_3d, level=0.0, spacing=(1.0, 1.0, 1.0)):
    verts, faces, _, _ = marching_cubes(sdf_3d, level=level, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.smooth()
    return mesh

# ===================== Streamlit 页面主逻辑 =====================
def main():
    st.title("SDF-SAM 框选区域8切片推理 + 3D重建")
    st.sidebar.header("功能菜单")

    # 加载模型
    model = load_model(MODEL_WEIGHT)
    st.sidebar.success("✅ 模型加载完成")

    # 上传NIfTI文件
    uploaded_file = st.sidebar.file_uploader("上传 .nii.gz 3D医学影像", type=["nii.gz"])
    if not uploaded_file:
        st.info("请在左侧上传 3D NIfTI 影像文件")
        return

    # 读取3D数据
    nii_img = nib.Nifti1Image(uploaded_file.read(), np.eye(4))
    img_data = nii_img.get_fdata()
    img_vol = img_data.transpose((2, 0, 1))  # (D, H, W)
    D, H, W = img_vol.shape
    st.subheader(f"影像信息: 总切片数={D}, 高={H}, 宽={W}")

    # 选择参考切片（用于画框）
    slice_idx = st.slider("选择参考切片（在此切片上手动画框）", 0, D-1, D//2)
    show_slice = img_vol[slice_idx]
    show_slice_norm = (show_slice - show_slice.min()) / (show_slice.max() + 1e-8)
    show_slice_uint8 = (show_slice_norm * 255).astype(np.uint8)

    # 交互式矩形画框
    st.subheader("🔲 在目标器官上拖拽绘制矩形框")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=show_slice_uint8,
        height=512,
        width=512,
        drawing_mode="rect",
        key="bbox_canvas"
    )

    # 解析框坐标
    rel_bbox = None
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            rect = objects[0]
            # 画布512尺寸 -> 归一化0~1
            cx = rect["left"] / 512
            cy = rect["top"] / 512
            cw = rect["width"] / 512
            ch = rect["height"] / 512
            xmin = cx
            ymin = cy
            xmax = cx + cw
            ymax = cy + ch
            rel_bbox = [xmin, ymin, xmax, ymax]
            st.success(f"已完成框选，归一化坐标: {rel_bbox}")
        else:
            st.warning("请在图像上拖拽鼠标绘制矩形框")

    # 执行推理+重建
    if st.button("提取8张切片并推理重建") and rel_bbox is not None:
        with st.spinner("1. 在框选范围内均匀提取8张有效切片..."):
            sample_ids, valid_ids = get_slices_in_bbox(img_vol, rel_bbox, H, W, NUM_SLICES)
            st.info(f"有效切片总数: {len(valid_ids)} | 已采样切片索引: {list(sample_ids)}")

        with st.spinner("2. 8张切片批量推理SDF..."):
            slice_sdf_dict = infer_8_slices_batch(model, img_vol, sample_ids, rel_bbox, H, W)

        with st.spinner("3. 补全全体积3D SDF场..."):
            full_sdf_3d = fill_3d_sdf(img_vol, slice_sdf_dict)

        with st.spinner("4. 重建3D器官表面网格..."):
            mesh = sdf_to_mesh(full_sdf_3d, level=0.0)

        st.success("全部流程执行完毕！")

        # 网格信息
        st.subheader("网格基本信息")
        st.write(f"顶点数量: {len(mesh.vertices)}")
        st.write(f"三角面片数量: {len(mesh.faces)}")

        # 下载网格文件
        mesh.export("./reconstructed_organ.obj")
        with open("./reconstructed_organ.obj", "rb") as f:
            st.download_button(
                label="下载3D网格 (.obj)",
                data=f,
                file_name="organ_mesh.obj",
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()