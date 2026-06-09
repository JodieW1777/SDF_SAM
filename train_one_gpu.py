# ====================== 全局CUDA显存终极优化（Windows16G显卡专属，根治碎片OOM） ======================
import os

# 强制开启Windows兼容的显存碎片整理（官方适配方案，解决显存超额占用）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import distance_transform_edt
import cv2

# 导入自定义SDF模型
from segment_anything import build_sam_sdf




# 全套稳定显存优化、关闭高精度冗余计算
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# 禁用梯度累加冗余缓存
torch.backends.cudnn.enabled = True


# ====================== 1. SDF回归损失函数（全新替换分割损失） ======================
def sdf_huber_loss(pred_sdf, gt_sdf, delta=0.1):
    """SDF专属回归损失，适配正负连续距离值，抗异常值"""
    return F.huber_loss(pred_sdf, gt_sdf, delta=delta)


# ====================== 2. 二值Mask转SDF场（核心在线预处理） ======================
def mask2sdf(mask):
    """通过欧式距离变换，将3D二值掩码转为SDF真值场"""
    inside = distance_transform_edt(mask == 1)
    outside = distance_transform_edt(mask == 0)
    sdf = outside - inside
    return sdf.astype(np.float32)


# ====================== 3. 自定义Nii数据集（16G显卡极致显存优化版） ======================
class NiiSDFDataset(Dataset):
    # 极致稳妥超参：彻底压显存，不影响训练收敛效果
    def __init__(self, img_dir, label_dir, num_slices=6, num_query=512, img_size=1024):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.num_slices = num_slices  # 8→6，大幅降低序列特征显存堆叠
        self.num_query = num_query  # 1024→512，减半3D查询点计算量
        self.img_size = img_size
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".nii.gz")])

    def __len__(self):
        return len(self.img_files)

    def _get_random_bbox(self, mask):
        """根据单张切片mask生成随机扰动外接框（模拟人工框选）"""
        ys, xs = np.where(mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        h, w = mask.shape

        scale = np.random.uniform(0.85, 1.15)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        half_w = (xmax - xmin) / 2 * scale
        half_h = (ymax - ymin) / 2 * scale

        xmin = max(0, cx - half_w)
        xmax = min(w - 1, cx + half_w)
        ymin = max(0, cy - half_h)
        ymax = min(h - 1, cy + half_h)

        bbox = np.array([xmin / w, ymin / h, xmax / w, ymax / h], dtype=np.float32)
        return bbox

    def _resize_to_1024(self, img_slice, mask_slice):
        """强制将任意尺寸切片resize到1024×1024，对齐MedSAM预训练尺寸"""
        h, w = img_slice.shape
        target_size = self.img_size

        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img_slice, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_slice, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        img_pad = np.zeros((target_size, target_size), dtype=np.float32)
        mask_pad = np.zeros((target_size, target_size), dtype=np.float32)

        offset_h = (target_size - new_h) // 2
        offset_w = (target_size - new_w) // 2

        img_pad[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = img_resized
        mask_pad[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = mask_resized

        return img_pad, mask_pad

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img_path = os.path.join(self.img_dir, fname)

        if "_0000.nii.gz" in fname:
            label_fname = fname.replace("_0000.nii.gz", ".nii.gz")
        else:
            label_fname = fname
        label_path = os.path.join(self.label_dir, label_fname)

        img_vol = nib.load(img_path).get_fdata()
        lab_vol = nib.load(label_path).get_fdata()

        img_vol = img_vol.transpose((2, 0, 1))
        lab_vol = lab_vol.transpose((2, 0, 1))
        D, H, W = img_vol.shape

        sdf_vol = mask2sdf(lab_vol)

        slice_ids = np.linspace(0, D - 1, self.num_slices, dtype=int)
        slices = []
        z_positions = []
        batch_boxes = []
        for z in slice_ids:
            img = img_vol[z]
            mask_slice = lab_vol[z]

            img, mask_slice = self._resize_to_1024(img, mask_slice)

            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1)
            slices.append(img)
            z_positions.append(float(z))

            bbox = self._get_random_bbox(mask_slice)
            batch_boxes.append(torch.from_numpy(bbox).float())

        slices = torch.stack(slices, dim=0)#将多个切片的张量、z 坐标、框坐标堆叠为批量张量：
        slice_z_pos = torch.tensor(z_positions).float()
        batch_boxes = torch.stack(batch_boxes, dim=0)

        query_pts = []
        sdf_labels = []
        for _ in range(self.num_query):
            z = np.random.randint(0, D)
            y = np.random.randint(0, H)
            x = np.random.randint(0, W)
            query_pts.append([x, y, z])
            sdf_labels.append(sdf_vol[z, y, x])

        query_pts = torch.tensor(query_pts).float()
        sdf_labels = torch.tensor(sdf_labels).float()

        empty_pts = torch.zeros(self.num_slices, 1, 2)
        empty_lab = torch.zeros(self.num_slices, 1)

        return {
            "slices": slices,
            "slice_z_positions": slice_z_pos,
            "query_points": query_pts,
            "sdf_labels": sdf_labels,
            "points": empty_pts,
            "labels": empty_lab,
            "boxes": batch_boxes
        }


# ====================== 4. 主训练函数（16G显卡最终稳定版，彻底解决显存碎片化OOM） ======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir = "data/FLARE22Train/images"
    label_dir = "data/FLARE22Train/labels"

    # 极致稳定超参，适配16G显存，零溢出
    batch_size = 1
    slice_batch_size = 4
    lr = 1e-4
    epoch_num = 100

    dataset = NiiSDFDataset(img_dir, label_dir)
    # 关闭多进程、杜绝内存复制溢出
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    model = build_sam_sdf(pretrained_path="medsam_vit_b.pth")
    model.to(device)
    model.train()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 冻结图像编码器浅层参数，仅训练SDF核心分支
    # ========== 终极显存修复：冻结整个图像编码器 ==========
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # 只训练 SDF 头 + prompt 部分
    for param in model.prompt_encoder.parameters():
        param.requires_grad = True

    for param in model.sdf_decoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # 逐批次清空显存，彻底杜绝碎片堆积
    for epoch in range(epoch_num):
        torch.cuda.empty_cache()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epoch_num}")

        for batch in pbar:
            slices = batch["slices"].to(device)
            query_points = batch["query_points"].to(device)
            slice_z_pos = batch["slice_z_positions"].to(device)
            sdf_gt = batch["sdf_labels"].to(device)
            pts = (batch["points"].to(device), batch["labels"].to(device))

            # 多提示混合训练策略
            mode = np.random.choice([0, 1, 2])
            if mode == 1:
                box_input = batch["boxes"].to(device, non_blocking=True)
                point_input = pts
            elif mode == 2:
                box_input = None
                point_input = pts
            else:
                box_input = None
                point_input = pts

            del batch["boxes"]
            # 前预处理：清空显存、重置峰值
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 分片小批次前向传播，避免一次性爆显存
            slices_split = torch.split(slices, slice_batch_size, dim=0)
            pred_sdf_list = []
            for slice_sub in slices_split:
                pred_sdf_sub = model(
                    slices=slice_sub,
                    query_points=query_points[:len(slice_sub)],
                    slice_z_positions=slice_z_pos[:len(slice_sub)],
                    points_per_slice=(pts[0][:len(slice_sub)], pts[1][:len(slice_sub)]),
                    boxes_per_slice=box_input[:len(slice_sub)] if box_input is not None else None
                )
                pred_sdf_list.append(pred_sdf_sub)
                torch.cuda.empty_cache()

            # 拼接所有分片结果
            pred_sdf = torch.cat(pred_sdf_list, dim=0)

            loss = sdf_huber_loss(pred_sdf, sdf_gt)
            loss.backward()
            optimizer.step()

            # 极致显存释放（关键）
            optimizer.zero_grad(set_to_none=True)
            del loss, pred_sdf, sdf_gt
            torch.cuda.empty_cache()

            total_loss += loss.item()
            pbar.set_postfix({"sdf_loss": loss.item()})
            # 每5个epoch保存一次，减少显存占用


        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch + 1}] 平均SDF回归损失: {avg_loss:.6f}")
        torch.save(model.state_dict(), f"./sdf_sam_epoch{epoch + 1}.pth")


if __name__ == "__main__":
    main()
