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
    def __init__(self, img_dir, label_dir, num_slices=4, num_query=1024, sdf_trunc=10.0):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.num_slices = num_slices
        self.num_query = num_query
        self.sdf_trunc = float(sdf_trunc)

        # 关键：预构建【(图像路径,标签路径,器官ID)】全列表，每个器官单独一条数据
        self.sample_list = []
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".nii.gz")])
        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            if "_0000.nii.gz" in fname:
                label_fname = fname.replace("_0000.nii.gz", ".nii.gz")
            else:
                label_fname = fname
            label_path = os.path.join(self.label_dir, label_fname)

            # 读取标签，提取该病例全部存在的器官
            lab_data = nib.load(label_path).get_fdata()
            lab_vol = lab_data.transpose((2, 0, 1))  # D,H,W
            organ_ids = np.unique(lab_vol)
            organ_ids = organ_ids[organ_ids != 0]  # 去掉背景0
            if len(organ_ids) == 0:
                continue
            # 当前CT每一个器官都存入样本池，不丢弃、不随机
            for oid in organ_ids:
                self.sample_list.append((img_path, label_path, int(oid)))
        print(f"数据集加载完成，总器官样本数量：{len(self.sample_list)}")

    def __len__(self):
        # 长度 = 所有CT所有器官总数
        return len(self.sample_list)

    @staticmethod
    def _prepare_native_slice(img_slice):
        """Normalize a slice without changing its native spatial grid."""
        img = np.asarray(img_slice, dtype=np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)

    def __getitem__(self, idx):
        # 直接根据idx取唯一一对：图像、标签、固定器官ID，无随机
        img_path, label_path, target_organ_id = self.sample_list[idx]

        img_data = nib.load(img_path).get_fdata()
        lab_data = nib.load(label_path).get_fdata()
        img_vol = img_data.transpose((2, 0, 1))  # D,H,W
        lab_vol = lab_data.transpose((2, 0, 1))
        D, H, W = img_vol.shape

        single_mask = (lab_vol == target_organ_id).astype(np.float32)
        # 单器官包围盒
        # 1. 获取器官所有坐标，得到包围盒
        z_coords, y_coords, x_coords = np.where(single_mask > 0)
        # The crop needs enough exterior context to define a useful positive
        # distance field. Two voxels made the box boundary dominate training.
        pad = int(np.ceil(self.sdf_trunc))
        xmin = max(0, x_coords.min() - pad)
        xmax = min(W - 1, x_coords.max() + pad)
        ymin = max(0, y_coords.min() - pad)
        ymax = min(H - 1, y_coords.max() + pad)
        zmin = max(0, z_coords.min() - pad)
        zmax = min(D - 1, z_coords.max() + pad)

        crop_mask = single_mask[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
        crop_sdf = mask2sdf(crop_mask)

        # A fixed TSDF scale is comparable across organs/cases. The zero level
        # remains the anatomical surface and |SDF| is bounded for stable loss.
        crop_sdf = np.clip(crop_sdf, -self.sdf_trunc, self.sdf_trunc)
        crop_sdf = crop_sdf / self.sdf_trunc
        sdf_vol = np.ones_like(single_mask, dtype=np.float32)
        sdf_vol[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1] = crop_sdf







        # Boxes remain normalized, so the same physical extent is preserved
        # for native XY/XZ/YZ planes of different pixel sizes.
        # ---------- 1 XY平面框（轴位切片：横轴x，纵轴y，图尺寸 W,H）----------
        box_xy = np.array(
            [xmin / W, ymin / H, xmax / W, ymax / H], dtype=np.float32
        )

        # ---------- 2 XZ平面框（冠状切片：横轴x，纵轴z，图尺寸 W,D）----------
        box_xz = np.array(
            [xmin / W, zmin / D, xmax / W, zmax / D], dtype=np.float32
        )

        # ---------- 3 YZ平面框（矢状切片：横轴y，纵轴z，图尺寸 H,D）----------
        box_yz = np.array(
            [ymin / H, zmin / D, ymax / H, zmax / D], dtype=np.float32
        )
        # ===================== XY 轴位切片（沿Z） =====================
        # Sparse slice positions must bracket the complete query bbox, including
        # its exterior TSDF context. Selecting only organ-positive slices left
        # padded query points outside the observed slice range.
        z_ids = np.rint(np.linspace(zmin, zmax, self.num_slices)).astype(int)

        xy_slices = []
        xy_rel_z = []
        xy_boxes = []  # XY每组切片对应XY平面投影框
        for z in z_ids:
            z = int(z)
            img = img_vol[z]
            xy_slices.append(self._prepare_native_slice(img))
            # Query points are expressed in voxel coordinates, so slice
            # positions must use the same coordinate system.
            xy_rel_z.append(float(z))
            xy_boxes.append(torch.from_numpy(box_xy))
        xy_slices = torch.stack(xy_slices)
        xy_rel = torch.tensor(xy_rel_z).float()
        xy_boxes = torch.stack(xy_boxes)

        # ===================== XZ 冠状切片（沿Y） =====================
        y_ids = np.rint(np.linspace(ymin, ymax, self.num_slices)).astype(int)

        xz_slices = []
        xz_rel_y = []
        xz_boxes = []  # XZ每组切片对应XZ平面投影框
        for y in y_ids:
            y = int(y)
            slice_2d = img_vol[:, y, :]  # D,W
            xz_slices.append(self._prepare_native_slice(slice_2d))
            xz_rel_y.append(float(y))
            xz_boxes.append(torch.from_numpy(box_xz))
        xz_slices = torch.stack(xz_slices)
        xz_rel = torch.tensor(xz_rel_y).float()
        xz_boxes = torch.stack(xz_boxes)

        # ===================== YZ 矢状切片（沿X） =====================
        x_ids = np.rint(np.linspace(xmin, xmax, self.num_slices)).astype(int)

        yz_slices = []
        yz_rel_x = []
        yz_boxes = []  # YZ每组切片对应YZ平面投影框
        for x in x_ids:
            x = int(x)
            slice_2d = img_vol[:, :, x]  # D,H
            yz_slices.append(self._prepare_native_slice(slice_2d))
            yz_rel_x.append(float(x))
            yz_boxes.append(torch.from_numpy(box_yz))
        yz_slices = torch.stack(yz_slices)
        yz_rel = torch.tensor(yz_rel_x).float()
        yz_boxes = torch.stack(yz_boxes)

        # ========== 替换原来全部np.where全局采样逻辑 ==========
        # 只在器官包围盒范围内采样，避免全局千万体素分配大数组
        # 包围盒边界
        z0, z1 = zmin, zmax
        y0, y1 = ymin, ymax
        x0, x1 = xmin, xmax

        # Stratified sampling with exact counts. Half the queries are kept in
        # a two-voxel band around the surface, where reconstruction matters.
        total_target = self.num_query
        surf_num = total_target // 2
        inner_num = total_target // 4
        outer_num = total_target - surf_num - inner_num
        local_sdf = sdf_vol[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]

        def sample_region(region_mask, count):
            coords = np.argwhere(region_mask)
            if len(coords) == 0:
                coords = np.argwhere(np.ones_like(region_mask, dtype=bool))
            chosen = coords[np.random.choice(len(coords), size=count, replace=len(coords) < count)]
            return chosen

        surface = sample_region(np.abs(local_sdf) <= (2.0 / self.sdf_trunc), surf_num)
        inside = sample_region(local_sdf < 0, inner_num)
        outside = sample_region(local_sdf > 0, outer_num)
        local_pts = np.concatenate([surface, inside, outside], axis=0)
        np.random.shuffle(local_pts)
        global_zyx = local_pts + np.array([z0, y0, x0], dtype=np.int64)
        sdf_labels = sdf_vol[global_zyx[:, 0], global_zyx[:, 1], global_zyx[:, 2]]
        query_pts = global_zyx[:, [2, 1, 0]].astype(np.float32)

        del local_sdf, surface, inside, outside, local_pts, global_zyx
        import gc
        gc.collect()
        query_pts = torch.from_numpy(query_pts)
        # print(
        #     "query range:",
        #     query_pts.min(dim=0)[0],
        #     query_pts.max(dim=0)[0]
        # )
        sdf_labels = torch.from_numpy(np.asarray(sdf_labels, dtype=np.float32))

        empty_pts = torch.zeros(self.num_slices, 1, 2)
        empty_lab = torch.zeros(self.num_slices, 1)


        return {
            # XY轴位
            "xy_slices": xy_slices,
            "xy_rel": xy_rel,
            "xy_boxes": xy_boxes,
            # XZ冠状
            "xz_slices": xz_slices,
            "xz_rel": xz_rel,
            "xz_boxes": xz_boxes,
            # YZ矢状
            "yz_slices": yz_slices,
            "yz_rel": yz_rel,
            "yz_boxes": yz_boxes,

            "query_points": query_pts,
            "sdf_labels": sdf_labels,
            "eikonal_target": torch.tensor(1.0 / self.sdf_trunc, dtype=torch.float32),
            "volume_size": torch.tensor([W, H, D], dtype=torch.float32),
            "points": empty_pts,
            "labels": empty_lab,

        }

# ====================== 4. 主训练函数 ======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir = "data/FLARE22Train/images"
    label_dir = "data/FLARE22Train/labels"
    batch_size = 1
    slice_batch_size = 12
    lr = 1e-5  # batch=1且新增模块较多，避免不同器官间正负偏置来回震荡
    epoch_num = 50
    dataset = NiiSDFDataset(img_dir, label_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    # Never freeze a randomly initialized ViT. This is the MedSAM ViT-B
    # checkpoint shipped with the project; only compatible encoder/prompt
    # weights are loaded and the new SDF/cross-slice modules start fresh.
    model = build_sam_sdf(pretrained_path="save")
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

    log_file = open("./txt_data/train_log_slice_24_plane.txt", "a", encoding="utf-8")
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
            xy = batch["xy_slices"].to(device)
            xz = batch["xz_slices"].to(device)
            yz = batch["yz_slices"].to(device)
            xy_rel = batch["xy_rel"].to(device)
            xz_rel = batch["xz_rel"].to(device)
            yz_rel = batch["yz_rel"].to(device)
            xy_box = batch["xy_boxes"].to(device)
            xz_box = batch["xz_boxes"].to(device)
            yz_box = batch["yz_boxes"].to(device)

            query_points = batch["query_points"].to(device)
            sdf_gt = batch["sdf_labels"].to(device)
            eikonal_target = batch["eikonal_target"].to(device)
            volume_size = batch["volume_size"].to(device)
            pts = (batch["points"].to(device), batch["labels"].to(device))

            torch.cuda.reset_peak_memory_stats()
            slices_split = torch.split(xy, slice_batch_size, dim=0)
            xy_split = torch.split(xy, slice_batch_size, dim=0)
            xz_split = torch.split(xz, slice_batch_size, dim=0)
            yz_split = torch.split(yz, slice_batch_size, dim=0)
            xy_rel_split = torch.split(xy_rel, slice_batch_size, dim=0)
            xz_rel_split = torch.split(xz_rel, slice_batch_size,dim=0)
            yz_rel_split = torch.split(yz_rel, slice_batch_size,dim=0)
            xy_box_split = torch.split(xy_box, slice_batch_size, dim=0)
            xz_box_split = torch.split(xz_box, slice_batch_size, dim=0)
            yz_box_split = torch.split(yz_box, slice_batch_size, dim=0)
            q_split = torch.split(query_points, slice_batch_size, dim=0)
            gt_split = torch.split(sdf_gt, slice_batch_size, dim=0)
            pred_sdf_list = []
            hub_sum = 0.0
            eik_sum = 0.0
            batch_loss_sum = 0.0
            mae_sum = 0.0
            sign_balance_sum = 0.0
            pts_coords, pts_labels = pts
            coords_split = torch.split(pts_coords, slice_batch_size, dim=0)
            lab_split = torch.split(pts_labels, slice_batch_size, dim=0)
            for slice_sub, xz_sub, yz_sub, \
                    xy_rel_sub, xz_rel_sub, yz_rel_sub, \
                    xy_box_sub, xz_box_sub, yz_box_sub, \
                    coord_sub, lab_sub, \
                    q_sub, gt_sub in zip(
                xy_split, xz_split, yz_split,
                xy_rel_split, xz_rel_split, yz_rel_split,
                xy_box_split, xz_box_split, yz_box_split,
                coords_split, lab_split,
                q_split, gt_split
            ):
                # 前向预测SDF
                pred_sdf_sub = model(
                    xy_slices=slice_sub,
                    xz_slices=xz_sub,
                    yz_slices=yz_sub,
                    xy_rel=xy_rel_sub,
                    xz_rel=xz_rel_sub,
                    yz_rel=yz_rel_sub,
                    xy_boxes=xy_box_sub,
                    xz_boxes=xz_box_sub,
                    yz_boxes=yz_box_sub,
                    query_points=q_sub,
                    points_per_slice=None
                )

                # Fixed-scale TSDF regression. A global term keeps the three
                # strata calibrated; the surface term gives the zero level
                # additional weight without triple-counting every group.
                gt_inner = gt_sub < 0
                gt_outer = gt_sub > 0
                gt_surface = torch.abs(gt_sub) <= 0.2
                loss_reg = F.smooth_l1_loss(pred_sdf_sub, gt_sub, beta=0.1)
                loss_surface = F.smooth_l1_loss(
                    pred_sdf_sub[gt_surface], gt_sub[gt_surface], beta=0.05
                )
                loss_hub = loss_reg + 2.0 * loss_surface

                # ========== 符号对齐损失 ==========
                # Balanced occupancy supervision prevents the field collapsing
                # to one sign. Positive logits represent exterior voxels.
                exterior_target = (gt_sub > 0).to(pred_sdf_sub.dtype)
                sign_logits = pred_sdf_sub / 0.25
                inside_mask = ~exterior_target.bool()
                outside_mask = exterior_target.bool()
                sign_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        sign_logits[inside_mask], exterior_target[inside_mask]
                    )
                    + F.binary_cross_entropy_with_logits(
                        sign_logits[outside_mask], exterior_target[outside_mask]
                    )
                )

                # ========== 正负均衡损失 ==========
                pos_ratio = torch.sigmoid(sign_logits).mean()
                sign_balance_loss = ((pos_ratio - 0.5) ** 2)

                # ========== 全局均值偏移约束（防止全正/全负） ==========
                mean_pred = torch.mean(pred_sdf_sub)
                mean_shift_loss = (mean_pred ** 2)

                # Finite-difference Eikonal regularization. This follows the
                # complete prediction path (including grid_sample) while only
                # requiring first-order autograd, which PyTorch supports.
                eps = 1.0
                # TSDF is constant in the saturated |target|=1 region, where
                # its correct gradient is zero. Applying ||grad||=1/trunc
                # there would directly conflict with TSDF regression. Restrict
                # Eikonal to an unsaturated narrow band and leave one finite-
                # difference step of margin (0.1 for truncation=10 voxels).
                if q_sub.shape[0] != 1:
                    raise ValueError("Eikonal narrow-band sampling currently requires batch_size=1")
                eik_candidates = torch.nonzero(
                    torch.abs(gt_sub[0]) < 0.9, as_tuple=False
                ).squeeze(1)
                if eik_candidates.numel() == 0:
                    raise RuntimeError("No unsaturated TSDF samples available for Eikonal loss")
                eik_count = min(256, eik_candidates.numel())
                eik_order = torch.randperm(eik_candidates.numel(), device=q_sub.device)[:eik_count]
                eik_ids = eik_candidates[eik_order]
                q_eik = q_sub[:, eik_ids]
                fd_queries = []
                max_xyz = (volume_size - 1.0).view(volume_size.shape[0], 1, 3)
                for axis in range(3):
                    offset = torch.zeros_like(q_eik)
                    offset[..., axis] = eps
                    fd_queries.append(torch.minimum(q_eik + offset, max_xyz))
                    fd_queries.append(torch.maximum(q_eik - offset, torch.zeros_like(q_eik)))
                q_fd = torch.cat(fd_queries, dim=1)
                pred_fd = model(
                    xy_slices=slice_sub,
                    xz_slices=xz_sub,
                    yz_slices=yz_sub,
                    xy_rel=xy_rel_sub,
                    xz_rel=xz_rel_sub,
                    yz_rel=yz_rel_sub,
                    xy_boxes=xy_box_sub,
                    xz_boxes=xz_box_sub,
                    yz_boxes=yz_box_sub,
                    query_points=q_fd,
                    points_per_slice=None
                )
                n_query = q_eik.shape[1]
                fd_parts = pred_fd.split(n_query, dim=1)
                grad = torch.stack(
                    [(fd_parts[2 * a] - fd_parts[2 * a + 1]) / (2.0 * eps) for a in range(3)],
                    dim=-1,
                )
                grad_norm = torch.linalg.vector_norm(grad, dim=-1)
                target_grad = eikonal_target.view(-1, 1)
                loss_eik = torch.mean((grad_norm - target_grad) ** 2)

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
                    + sign_loss * 0.5
                    + sign_balance_loss * 0.1
                    # + mean_shift_loss* 0.8
                    + loss_eik * 0.1
                )


                # 梯度裁剪，防止爆炸震荡
                # torch_nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                batch_loss_sum += batch_total.item()
                batch_total.backward()

                # 指标统计
                batch_mae = torch.mean(torch.abs(pred_sdf_sub - gt_sub)).item()
                mae_sum += batch_mae
                hub_sum += loss_hub.item()
                eik_sum += loss_eik.item()
                sign_balance_sum += sign_balance_loss.item()
                pred_sdf_list.append(pred_sdf_sub.detach())

                # Release the large per-query tensors. Keep this list in sync
                # with the active loss implementation (sign_penalty belonged to
                # the removed legacy sign loss and caused an UnboundLocalError).
                del pred_sdf_sub, batch_total
                del loss_reg, loss_surface, loss_hub, sign_loss, sign_logits
                del exterior_target, inside_mask, outside_mask
                del loss_eik, sign_balance_loss, mean_pred, mean_shift_loss
                del q_eik, eik_ids, eik_order, eik_candidates
                del q_fd, pred_fd, fd_queries, fd_parts
                del grad, grad_norm
                del gt_inner, gt_outer, gt_surface, pos_ratio

            # Clip after all backward calls and immediately before the update.
            # The previous commented placement was before backward and thus
            # would not have clipped anything.
            grad_norm_total = torch_nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=1.0
            )
            optimizer.step()
            pred_sdf = torch.cat(pred_sdf_list, dim=0)
            batch_mae_avg = mae_sum / len(slices_split)
            batch_hub_avg = hub_sum / len(slices_split)
            batch_eik_avg = eik_sum / len(slices_split)
            batch_sign_balance_avg = sign_balance_sum / len(slices_split)
            total_batch_loss = batch_loss_sum / len(slices_split)
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
            torch.save(model.state_dict(), "result/best_sdf_sam_slice_24_plane.pth")
            print(f"已更新最优模型，最优损失: {best_loss:.6f}")
        # 每10轮保存
        # if epoch % 10 == 0:
        os.makedirs("./save_path/slice_24_plane", exist_ok=True)
        torch.save(model.state_dict(), f"./save_path/slice_24_plane/sdf_sam_epoch{epoch + 1}.pth")

    # 绘制训练曲线
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, loss_list, "r-", label="Train Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练总损失曲线")
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
    plt.savefig("train_curve_slice_24_plane.png")
    plt.show()
    log_file.close()
    print("训练完成！日志:train_log_slice_24_plane 曲线图已保存")

if __name__ == "__main__":
    main()
