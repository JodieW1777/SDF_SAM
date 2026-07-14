import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import numpy as np
import torch
import torch.nn as torch_nn
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

# ====================== 2. 二值Mask转SDF场（核心在线预处理） ======================
def mask2sdf(mask):
    """通过欧式距离变换，将3D二值掩码转为SDF真值场"""
    inside = distance_transform_edt(mask == 1)
    outside = distance_transform_edt(mask == 0)
    sdf = outside - inside
    return sdf.astype(np.float32)

# ====================== 3. 自定义Nii数据集 ======================
class NiiSDFDataset(Dataset):
    def __init__(self, img_dir, label_dir, num_slices=8, num_query=1024, img_size=1024):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.num_slices = num_slices
        self.num_query = num_query
        self.img_size = img_size
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".nii.gz")])
    def __len__(self):
        return len(self.img_files)

    def _resize_to_1024(self, img_slice, mask_slice):
        """强制将任意尺寸切片resize到1024×1024"""
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
        # 转维度: (D, H, W)
        img_vol = img_vol.transpose((2, 0, 1))
        lab_vol = lab_vol.transpose((2, 0, 1))
        D, H, W = img_vol.shape
        sdf_vol = mask2sdf(lab_vol)

        # ========== 修复：分内外独立归一，抹平数值量级差 ==========
        inner_mask = lab_vol > 0
        outer_mask = lab_vol == 0
        sdf_inner = sdf_vol[inner_mask]
        sdf_outer = sdf_vol[outer_mask]
        scale_inner = np.percentile(np.abs(sdf_inner), 95) + 1e-8
        scale_outer = np.percentile(np.abs(sdf_outer), 95) + 1e-8
        sdf_vol[inner_mask] /= scale_inner
        sdf_vol[outer_mask] /= scale_outer

        # ====================== 全局统一3D包围盒 ======================
        z_coords, y_coords, x_coords = np.where(lab_vol > 0)
        if len(z_coords) == 0:
            xmin, xmax = 0, W - 1
            ymin, ymax = 0, H - 1
            zmin, zmax = 0, D - 1
        else:
            pad = 10
            xmin = max(0, x_coords.min() - pad)
            xmax = min(W - 1, x_coords.max() + pad)
            ymin = max(0, y_coords.min() - pad)
            ymax = min(H - 1, y_coords.max() + pad)
            zmin = max(0, z_coords.min())
            zmax = min(D - 1, z_coords.max())

        # 筛选有效切片
        valid_slice_ids = []
        for z in range(D):
            mask_slice = lab_vol[z]
            if np.any(mask_slice > 0):
                valid_slice_ids.append(z)
        if len(valid_slice_ids) >= self.num_slices:
            slice_ids = np.linspace(0, len(valid_slice_ids) - 1, self.num_slices, dtype=int)
            slice_ids = np.array(valid_slice_ids)[slice_ids]
        else:
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
            # 全局统一框
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            half_w = (xmax - xmin) / 2
            half_h = (ymax - ymin) / 2
            bbox = np.array([
                (cx - half_w) / W,
                (cy - half_h) / H,
                (cx + half_w) / W,
                (cy + half_h) / H
            ], dtype=np.float32)
            batch_boxes.append(torch.from_numpy(bbox).float())
        slices = torch.stack(slices, dim=0)
        slice_z_pos = torch.tensor(z_positions).float()
        batch_boxes = torch.stack(batch_boxes, dim=0)

        # ====================== query 仅在全局3D包围盒内采样 ======================
        query_pts = []
        sdf_labels = []
        # 1. 近表面点 ±2
        coords = np.where(np.abs(sdf_vol) < 2)
        if len(coords[0]) > 0:
            sample_num = min(self.num_query // 2, len(coords[0]))
            idx = np.random.choice(len(coords[0]), size=sample_num, replace=True)
            for i in idx:
                x = coords[2][i]
                y = coords[1][i]
                z = coords[0][i]
                query_pts.append([x, y, z])
                sdf_labels.append(sdf_vol[z, y, x])
        # 2. 器官内部点
        coords = np.where(lab_vol > 0)
        if len(coords[0]) > 0:
            idx = np.random.choice(len(coords[0]), size=self.num_query // 4, replace=True)
            for i in idx:
                x = coords[2][i]
                y = coords[1][i]
                z = coords[0][i]
                query_pts.append([x, y, z])
                sdf_labels.append(sdf_vol[z, y, x])
        # 3. 外部点：限制在3D包围盒内，扩大重试次数
        count = self.num_query - len(query_pts)
        max_try = 100
        for _ in range(count):
            for _try in range(max_try):
                x = np.random.randint(xmin, xmax + 1)
                y = np.random.randint(ymin, ymax + 1)
                z = np.random.randint(zmin, zmax + 1)
                if lab_vol[z, y, x] == 0:
                    query_pts.append([x, y, z])
                    sdf_labels.append(sdf_vol[z, y, x])
                    break

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

# ====================== 4. 主训练函数 ======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir = "data/FLARE22Train/images"
    label_dir = "data/FLARE22Train/labels"
    batch_size = 1
    slice_batch_size = 8
    lr = 5e-5  # 下调学习率，多约束防止震荡
    epoch_num = 100
    dataset = NiiSDFDataset(img_dir, label_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    model = build_sam_sdf(pretrained_path="sdf_sam_epoch13.pth")
    model.to(device)
    model.train()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 冻结图像编码器
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.prompt_encoder.parameters():
        param.requires_grad = True
    for param in model.sdf_decoder.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    log_file = open("train_log_slice_cross_transformer_sin.txt", "a", encoding="utf-8")
    best_loss = float("inf")
    epoch_list = []
    loss_list = []
    mae_list = []
    LAMB_EIK = 0.1
    HUBER_DELTA = 0.1

    for epoch in range(epoch_num):
        total_loss = 0.0
        total_hub_loss = 0.0
        total_eik_loss = 0.0
        total_mae = 0.0
        total_sign_balance = 0.0
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
            q_split = torch.split(query_points, slice_batch_size, dim=0)
            gt_split = torch.split(sdf_gt, slice_batch_size, dim=0)
            pred_sdf_list = []
            hub_sum = 0.0
            eik_sum = 0.0
            mae_sum = 0.0
            sign_balance_sum = 0.0

            for idx, slice_sub in enumerate(slices_split):
                start_idx = idx * slice_batch_size
                end_idx = start_idx + len(slice_sub)
                q_sub = q_split[idx]
                gt_sub = gt_split[idx]

                # 前向预测SDF
                pred_sdf_sub = model(
                    slices=slice_sub,
                    query_points=q_sub,
                    slice_z_positions=slice_z_pos[start_idx:end_idx],
                    points_per_slice=(pts[0][start_idx:end_idx], pts[1][start_idx:end_idx]),
                    boxes_per_slice=box_input[start_idx:end_idx] if box_input is not None else None
                )

                # ========== 分内外加权Huber损失 ==========
                gt_inner = gt_sub < 0
                gt_outer = gt_sub > 0
                gt_surface = torch.abs(gt_sub) < 2.0
                loss_inner = F.huber_loss(pred_sdf_sub[gt_inner], gt_sub[gt_inner], delta=2.0)
                loss_surface = F.huber_loss(pred_sdf_sub[gt_surface], gt_sub[gt_surface], delta=2.0)
                loss_outer_raw = F.huber_loss(pred_sdf_sub[gt_outer], gt_sub[gt_outer], delta=2.0)
                loss_outer = 0.1 * loss_outer_raw
                loss_hub = loss_inner + loss_surface + loss_outer

                # ========== 符号对齐损失 ==========
                sign_penalty = torch.relu(-gt_sub * pred_sdf_sub)
                sign_loss = torch.mean(sign_penalty) * 0.8

                # ========== 0等值面约束 ==========
                zero_mask = torch.abs(gt_sub) < 0.1
                if zero_mask.any():
                    loss_zero = torch.mean(pred_sdf_sub[zero_mask] ** 2) * 0.5
                else:
                    loss_zero = torch.tensor(0.0, device=pred_sdf_sub.device)

                # ========== 正负均衡损失 ==========
                pos_ratio = torch.mean((pred_sdf_sub > 0).float())
                sign_balance_loss = ((pos_ratio - 0.5) ** 2) * 1.0

                # ========== 全局均值偏移约束（防止全正/全负） ==========
                mean_pred = torch.mean(pred_sdf_sub)
                mean_shift_loss = (mean_pred ** 2) * 0.3

                # ========== Eikonal梯度正则 ==========
                q_grad_in = q_sub.clone().detach().requires_grad_(True)
                pred_for_grad = model(
                    slices=slice_sub,
                    query_points=q_grad_in,
                    slice_z_positions=slice_z_pos[start_idx:end_idx],
                    points_per_slice=(pts[0][start_idx:end_idx], pts[1][start_idx:end_idx]),
                    boxes_per_slice=box_input[start_idx:end_idx] if box_input is not None else None
                )
                grad = torch.autograd.grad(
                    outputs=pred_for_grad.sum(),
                    inputs=q_grad_in,
                    create_graph=True,
                    retain_graph=True
                )[0]
                grad_norm = torch.norm(grad, dim=-1)
                loss_eik = torch.mean((grad_norm - 1.0) ** 2) * LAMB_EIK
                print(
                    "GT:",
                    sdf_gt.min().item(),
                    sdf_gt.max().item(),
                    (sdf_gt < 0).sum().item(),
                    (sdf_gt > 0).sum().item()
                )

                print(
                    "Pred:",
                    pred_sdf_sub.min().item(),
                    pred_sdf_sub.max().item(),
                    (pred_sdf_sub < 0).sum().item(),
                    (pred_sdf_sub > 0).sum().item()
                )
                # ========== 总损失 ==========
                batch_total = (
                    loss_hub
                    + sign_loss
                    + loss_zero
                    + sign_balance_loss
                    + mean_shift_loss
                    + loss_eik
                )


                # 梯度裁剪，防止爆炸震荡
                torch_nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                batch_total.backward()

                # 指标统计
                batch_mae = torch.mean(torch.abs(pred_sdf_sub - gt_sub)).item()
                mae_sum += batch_mae
                hub_sum += loss_hub.item()
                eik_sum += loss_eik.item()
                sign_balance_sum += sign_balance_loss.item()
                pred_sdf_list.append(pred_sdf_sub.detach())

            optimizer.step()
            pred_sdf = torch.cat(pred_sdf_list, dim=0)
            batch_mae_avg = mae_sum / len(slices_split)
            batch_hub_avg = hub_sum / len(slices_split)
            batch_eik_avg = eik_sum / len(slices_split)
            batch_sign_balance_avg = sign_balance_sum / len(slices_split)
            total_batch_loss = batch_hub_avg + batch_eik_avg
            total_loss += total_batch_loss
            total_hub_loss += batch_hub_avg
            total_eik_loss += batch_eik_avg
            total_mae += batch_mae
            total_sign_balance += batch_sign_balance_avg

            pbar.set_postfix({
                "total_loss": f"{total_batch_loss:.4f}",
                "hub": f"{batch_hub_avg:.4f}",
                "eik": f"{batch_eik_avg:.4f}",
                "mae": f"{batch_mae:.2f}",
                "sign_bal": f"{batch_sign_balance_avg:.4f}"
            })
            optimizer.zero_grad(set_to_none=True)
            del pred_sdf, sdf_gt

        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        epoch_list.append(epoch + 1)
        loss_list.append(avg_loss)
        mae_list.append(avg_mae)
        print(f"[Epoch {epoch + 1}] 平均总损失: {avg_loss:.6f} | 平均MAE: {avg_mae:.6f}")
        log_info = f"Epoch:{epoch+1:03d} TotalLoss:{avg_loss:.6f} MAE:{avg_mae:.6f}\n"
        log_file.write(log_info)
        log_file.flush()

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_sdf_sam_slice_cross_transformer_sin.pth")
            print(f"已更新最优模型，最优损失: {best_loss:.6f}")
        # 每10轮保存
        # if epoch % 10 == 0:
        os.makedirs("./save_path/slice_cross_transformer_sin", exist_ok=True)
        torch.save(model.state_dict(), f"./save_path/slice_cross_transformer_sin/sdf_sam_epoch{epoch + 1}.pth")

    # 绘制训练曲线
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, loss_list, "r-", label="Train Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练总损失曲线(Huber+Eikonal)")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, mae_list, "b-", label="Train MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("SDF平均绝对误差曲线")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_curve_slice_cross_transformer_sin.png")
    plt.show()
    log_file.close()
    print("训练完成！日志:train_log_slice_cross_transformer_sin 曲线图已保存")

if __name__ == "__main__":
    main()
