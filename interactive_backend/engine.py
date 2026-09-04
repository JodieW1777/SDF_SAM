from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes
import trimesh

from segment_anything import build_sam_sdf
@dataclass(frozen=True)
class ReconstructionResult:
    mesh: trimesh.Trimesh
    sdf_min: float
    sdf_max: float
    query_count: int
    grid_shape: tuple[int, int, int]


class MedSAMReconstructionEngine:
    """Loads the network once and serializes GPU inference requests."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = build_sam_sdf(pretrained_path=None)
        state = torch.load(str(checkpoint), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self._lock = threading.Lock()

    @staticmethod
    def _sample_axis_slices(volume, bbox, axis, num_slices):
        D, H, W = volume.shape
        x0, y0, z0, x1, y1, z1 = bbox
        if axis == 0:
            positions = np.rint(np.linspace(z0, z1, num_slices)).astype(int)
            plane_box = np.array([x0 / W, y0 / H, x1 / W, y1 / H], np.float32)
            get_slice = lambda a: volume[a, :, :]
        elif axis == 1:
            positions = np.rint(np.linspace(y0, y1, num_slices)).astype(int)
            plane_box = np.array([x0 / W, z0 / D, x1 / W, z1 / D], np.float32)
            get_slice = lambda a: volume[:, a, :]
        else:
            positions = np.rint(np.linspace(x0, x1, num_slices)).astype(int)
            plane_box = np.array([y0 / H, z0 / D, y1 / H, z1 / D], np.float32)
            get_slice = lambda a: volume[:, :, a]

        slices = []
        for position in positions:
            image = np.asarray(get_slice(int(position)), dtype=np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            slices.append(torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1))
        boxes = torch.from_numpy(np.repeat(plane_box[None], num_slices, axis=0))
        return torch.stack(slices), torch.tensor(positions, dtype=torch.float32), boxes

    @staticmethod
    def _clamp_bbox(bbox, shape_xyz):
        bbox = np.asarray(bbox, dtype=np.float64)
        lo = np.floor(np.minimum(bbox[:3], bbox[3:])).astype(int)
        hi = np.ceil(np.maximum(bbox[:3], bbox[3:])).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(shape_xyz) - 1)
        if np.any(hi <= lo):
            raise ValueError(f"bbox is empty after clamping: {lo.tolist()}..{hi.tolist()}")
        return [*lo.tolist(), *hi.tolist()]

    @staticmethod
    def _make_query_grid(bbox, budget):
        x0, y0, z0, x1, y1, z1 = bbox
        extents = np.array([x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1], dtype=np.float64)
        scale = min(1.0, (budget / np.prod(extents)) ** (1.0 / 3.0))
        counts = np.maximum(2, np.ceil(extents * scale).astype(int))
        xs = np.linspace(x0, x1, counts[0], dtype=np.float32)
        ys = np.linspace(y0, y1, counts[1], dtype=np.float32)
        zs = np.linspace(z0, z1, counts[2], dtype=np.float32)
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        return points, (xs, ys, zs)

    @torch.inference_mode()
    def _predict_queries(self, inputs, queries, chunk_size):
        predictions = []
        for start in range(0, len(queries), chunk_size):
            q = torch.from_numpy(queries[start:start + chunk_size]).to(self.device)
            pred = self.model(query_points=q.unsqueeze(0), points_per_slice=None, **inputs)
            predictions.append(pred.squeeze(0).float().cpu())
        return torch.cat(predictions).numpy()

    def reconstruct(
        self,
        image_path: str | Path,
        bbox_index,
        index_to_world,
        num_slices=8,
        query_budget=100_000,
        query_chunk_size=20_000,
        level=0.0,
    ) -> ReconstructionResult:
        nii = nib.load(str(image_path))
        image_xyz = np.asarray(nii.get_fdata(dtype=np.float32))
        if image_xyz.ndim != 3:
            raise ValueError(f"Only scalar 3-D images are supported, got {image_xyz.shape}")
        # Model convention is [D,H,W] == [z,y,x].
        volume = image_xyz.transpose(2, 1, 0)
        D, H, W = volume.shape
        bbox = self._clamp_bbox(bbox_index, (W, H, D))

        xy, xy_rel, xy_box = self._sample_axis_slices(volume, bbox, 0, num_slices)
        xz, xz_rel, xz_box = self._sample_axis_slices(volume, bbox, 1, num_slices)
        yz, yz_rel, yz_box = self._sample_axis_slices(volume, bbox, 2, num_slices)
        inputs = {
            "xy_slices": xy.unsqueeze(0).to(self.device),
            "xz_slices": xz.unsqueeze(0).to(self.device),
            "yz_slices": yz.unsqueeze(0).to(self.device),
            "xy_rel": xy_rel.unsqueeze(0).to(self.device),
            "xz_rel": xz_rel.unsqueeze(0).to(self.device),
            "yz_rel": yz_rel.unsqueeze(0).to(self.device),
            "xy_boxes": xy_box.unsqueeze(0).to(self.device),
            "xz_boxes": xz_box.unsqueeze(0).to(self.device),
            "yz_boxes": yz_box.unsqueeze(0).to(self.device),
        }
        queries, axes = self._make_query_grid(bbox, query_budget)
        with self._lock:
            values = self._predict_queries(inputs, queries, query_chunk_size)

        xs, ys, zs = axes
        coarse = values.reshape(len(zs), len(ys), len(xs))
        sdf_min, sdf_max = float(coarse.min()), float(coarse.max())
        if not sdf_min < level < sdf_max:
            raise RuntimeError(
                f"TSDF does not cross level {level}: min={sdf_min:.5f}, max={sdf_max:.5f}"
            )

        # Marching cubes runs on a dense local voxel grid. Regular-grid
        # interpolation is deterministic and hole-free unlike random griddata.
        x0, y0, z0, x1, y1, z1 = bbox
        xi = np.arange(x0, x1 + 1, dtype=np.float32)
        yi = np.arange(y0, y1 + 1, dtype=np.float32)
        zi = np.arange(z0, z1 + 1, dtype=np.float32)
        interpolator = RegularGridInterpolator(
            (zs, ys, xs), coarse, method="linear", bounds_error=False, fill_value=1.0
        )
        zz, yy, xx = np.meshgrid(zi, yi, xi, indexing="ij")
        dense = interpolator(np.stack([zz, yy, xx], axis=-1)).astype(np.float32)
        verts_zyx, faces, normals, _ = marching_cubes(dense, level=level)
        verts_zyx += np.array([z0, y0, x0], dtype=np.float32)
        verts_xyz = verts_zyx[:, [2, 1, 0]]
        # Return raw CT index coordinates. PLY has no standardized medical
        # RAS/LPS metadata, so the MITK client applies its own ImageGeometry
        # IndexToWorld transform after loading the PLY.
        mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)
        mesh.remove_unreferenced_vertices()
        return ReconstructionResult(mesh, sdf_min, sdf_max, len(queries), dense.shape)
