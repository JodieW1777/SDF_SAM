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
class NpySDFDataset(Dataset):
    def __init__(self, npy_root, num_slices=4, num_query=512):
        self.img_dir = os.path.join(npy_root, "imgs")
        self.gt_dir = os.path.join(npy_root, "gts")
        self.num_slices = num_slices
        self.num_query = num_query
        # 同步匹配图片+mask文件名
        self.img_names = sorted(os.listdir(self.img_dir))
        self.valid_pairs = []
        for name in self.img_names:
            gt_path = os.path.join(self.gt_dir, name)
            if os.path.exists(gt_path):
                self.valid_pairs.append(name)
        print(f"有效样本总数: {len(self.valid_pairs)}")

    def __len__(self):
        return len(self.valid_pairs)

    def _get_random_bbox(self, mask):
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
        bbox = np.array([xmin / w, ymin / h, xmax / w, ymax], dtype=np.float32)
        return bbox

    def __getitem__(self, idx):
        name = self.valid_pairs[idx]
        img_path = os.path.join(self.img_dir, name)
        gt_path = os.path.join(self.gt_dir, name)
        # pre_CT_MR输出固定(1024,1024,3) [0,1]
        img_slice = np.load(img_path)
        mask_slice = np.load(gt_path)

        # 这里模拟3D volume逻辑：单切片重复构造虚拟D维度用于SDF采样
        D = self.num_slices
        img_vol = np.repeat(img_slice[np.newaxis, ...], D, axis=0)  # D,1024,1024,3
        lab_vol = np.repeat(mask_slice[np.newaxis, ...], D, axis=0) # D,1024,1024
        sdf_vol = mask2sdf(lab_vol)

        # 均匀采样N张切片
        slice_ids = np.linspace(0, D-1, self.num_slices, dtype=int)
        slices = []
        z_positions = []
        batch_boxes = []
        for z in slice_ids:
            img = img_vol[z]
            img = torch.from_numpy(img).permute(2,0,1).float() # (3,1024,1024)
            slices.append(img)
            z_positions.append(float(z))
            bbox = self._get_random_bbox(mask_slice)
            batch_boxes.append(torch.from_numpy(bbox).float())

        slices = torch.stack(slices, dim=0)
        slice_z_pos = torch.tensor(z_positions).float()
        batch_boxes = torch.stack(batch_boxes, dim=0)

        # 随机3D查询点
        query_pts = []
        sdf_labels = []
        for _ in range(self.num_query):
            z = np.random.randint(0, D)
            y = np.random.randint(0, 1024)
            x = np.random.randint(0, 1024)
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
    # npy输出根目录，对应pre_CT_MR生成的./npy
    npy_root = "./data/npy/CT_Abd"
    batch_size = 1
    slice_batch_size = 4
    lr=1e-4
    epoch_num = 100
    dataset = NpySDFDataset(npy_root=npy_root, num_slices=4, num_query=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    model = build_sam_sdf(pretrained_path="medsam_vit_b.pth")
    model.to(device)
    model.train()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 冻结编码器
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.prompt_encoder.parameters():
        param.requires_grad = True
    for param in model.sdf_decoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # ========== 新增：日志、最优模型、绘图容器 ==========
    log_file = open("train_log_slice_cross_transformer.txt", "a", encoding="utf-8")
    best_loss = float("inf")
    epoch_list = []
    loss_list = []
    mae_list = []

    for epoch in range(epoch_num):
        total_loss = 0.0
        total_mae = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epoch_num}")
        for batch in pbar:
            slices = batch["slices"].to(device)
            query_points = batch["query_points"].to(device)
            slice_z_pos = batch["slice_z_positions"].to(device)
            sdf_gt = batch["sdf_labels"].to(device)
            pts = (batch["points"].to(device), batch["labels"].to(device))

            box_input = batch["boxes"].to(device, non_blocking=True)
            del batch["boxes"]

            torch.cuda.reset_peak_memory_stats()

            slices_split = torch.split(slices, slice_batch_size, dim=0)
            pred_sdf_list = []
            for idx, slice_sub in enumerate(slices_split):
                start_idx = idx * slice_batch_size
                end_idx = start_idx + len(slice_sub)
                pred_sdf_sub = model(
                    slices=slice_sub,
                    query_points=query_points[start_idx:end_idx],
                    slice_z_positions=slice_z_pos[start_idx:end_idx],
                    points_per_slice=(pts[0][start_idx:end_idx], pts[1][start_idx:end_idx]),
                    boxes_per_slice=box_input[start_idx:end_idx] if box_input is not None else None
                )
                pred_sdf_list.append(pred_sdf_sub)
            pred_sdf = torch.cat(pred_sdf_list, dim=0)
            loss = sdf_huber_loss(pred_sdf, sdf_gt)
            loss.backward()
            optimizer.step()

            # 计算MAE
            batch_mae = torch.mean(torch.abs(pred_sdf - sdf_gt)).item()
            total_mae += batch_mae

            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix({"sdf_loss": loss_val, "mae": batch_mae})

            optimizer.zero_grad(set_to_none=True)
            del loss, pred_sdf, sdf_gt, loss_val

        # 单轮epoch统计
        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        epoch_list.append(epoch+1)
        loss_list.append(avg_loss)
        mae_list.append(avg_mae)

        # 终端打印 + 写入日志
        print(f"[Epoch {epoch + 1}] 平均损失: {avg_loss:.6f} | 平均MAE: {avg_mae:.6f}")
        log_info = f"Epoch:{epoch+1:03d} Loss:{avg_loss:.6f} MAE:{avg_mae:.6f}\n"
        log_file.write(log_info)
        log_file.flush()

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_sdf_sam_slice_cross_transformer_sin.pth")
            print(f"已更新最优模型，最优损失: {best_loss:.6f}")
        # 保存每轮模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./save_path/slice_cross_transformer_sin/sdf_sam_epoch{epoch + 1}.pth")

    # 训练结束，绘制曲线
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epoch_list, loss_list, "r-", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练损失曲线")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epoch_list, mae_list, "b-", label="Train MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("SDF平均绝对误差曲线")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_curve_slice_cross_transformer.png")
    plt.show()

    log_file.close()
    print("训练完成！日志:train_log.txt  曲线图:train_curve.png")


if __name__ == "__main__":
    main()
