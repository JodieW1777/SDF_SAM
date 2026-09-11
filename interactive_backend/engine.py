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
    prompt_bbox: tuple[int, ...]
    model_prompt_bbox: tuple[int, ...]
    reconstruction_bbox: tuple[int, ...]
    mesh_bbox: tuple[float, ...]
    boundary_negative_ratio: tuple[float, ...]


class MedSAMReconstructionEngine:
    """Loads the network once and serializes GPU inference requests."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint = str(Path(checkpoint).resolve())
        self.model = build_sam_sdf(pretrained_path=None)
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self._lock = threading.Lock()

    @staticmethod
    def _sample_axis_slices(volume, slice_bbox, prompt_bbox, axis, num_slices):
        D, H, W = volume.shape
        x0, y0, z0, x1, y1, z1 = slice_bbox
        px0, py0, pz0, px1, py1, pz1 = prompt_bbox
        if axis == 0:
            positions = np.rint(np.linspace(z0, z1, num_slices)).astype(int)
            plane_box = np.array([px0 / W, py0 / H, px1 / W, py1 / H], np.float32)
            get_slice = lambda a: volume[a, :, :]
        elif axis == 1:
            positions = np.rint(np.linspace(y0, y1, num_slices)).astype(int)
            plane_box = np.array([px0 / W, pz0 / D, px1 / W, pz1 / D], np.float32)
            get_slice = lambda a: volume[:, a, :]
        else:
            positions = np.rint(np.linspace(x0, x1, num_slices)).astype(int)
            plane_box = np.array([py0 / H, pz0 / D, py1 / H, pz1 / D], np.float32)
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

    @staticmethod
    def _expand_bbox(bbox, margin, shape_xyz):
        lo = np.maximum(np.asarray(bbox[:3], dtype=np.int64) - margin, 0)
        hi = np.minimum(
            np.asarray(bbox[3:], dtype=np.int64) + margin,
            np.asarray(shape_xyz, dtype=np.int64) - 1,
        )
        return [*lo.tolist(), *hi.tolist()]

    @staticmethod
    def _transform_points(points_xyz, matrix):
        points_xyz = np.asarray(points_xyz, dtype=np.float64)
        homogeneous = np.concatenate(
            [points_xyz, np.ones((len(points_xyz), 1), dtype=np.float64)], axis=1
        )
        transformed = homogeneous @ np.asarray(matrix, dtype=np.float64).T
        return transformed[:, :3] / transformed[:, 3:4]

    @classmethod
    def _transform_bbox(cls, bbox, matrix):
        lo = np.asarray(bbox[:3], dtype=np.float64)
        hi = np.asarray(bbox[3:], dtype=np.float64)
        corners = np.array([
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ])
        transformed = cls._transform_points(corners, matrix)
        return [*transformed.min(axis=0).tolist(), *transformed.max(axis=0).tolist()]

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
        image_shape_xyz,
        num_slices=4,
        reconstruction_margin=10,
        query_budget=100_000,
        query_chunk_size=20_000,
        level=0.0,
    ) -> ReconstructionResult:
        nii = nib.load(str(image_path))
        image_xyz = np.asarray(nii.get_fdata(dtype=np.float32))
        if image_xyz.ndim != 3:
            raise ValueError(f"Only scalar 3-D images are supported, got {image_xyz.shape}")
        # IMPORTANT: keep the exact historical training/test.py convention.
        # Both use image_xyz.transpose((2, 0, 1)), hence the model's logical
        # coordinates are (model_x, model_y, model_z) = (NIfTI_y, NIfTI_x,
        # NIfTI_z). The first two source dimensions are commonly both 512, so
        # a wrong (2,1,0) transpose silently passes every shape check while
        # presenting transposed anatomy and mismatched prompts to the model.
        volume = image_xyz.transpose(2, 0, 1)
        D, H, W = volume.shape
        source_shape_xyz = tuple(map(int, image_xyz.shape))
        model_shape_xyz = (W, H, D)
        mitk_shape_xyz = tuple(map(int, image_shape_xyz))
        mitk_index_to_lps = np.asarray(index_to_world, dtype=np.float64).reshape(4, 4)
        if (not np.isfinite(mitk_index_to_lps).all()
                or abs(np.linalg.det(mitk_index_to_lps[:3, :3])) < 1e-12):
            raise ValueError("MITK index-to-world transform is invalid or singular")

        if mitk_shape_xyz != source_shape_xyz:
            raise ValueError(
                "MITK image dimensions and exported NIfTI array differ: "
                f"MITK={mitk_shape_xyz}, NIfTI={source_shape_xyz}. "
                "Refusing to mix their index coordinates."
            )
        prompt_bbox = self._clamp_bbox(bbox_index, mitk_shape_xyz)
        # Convert the MITK/NIfTI source index bbox into the model convention
        # used by training and test.py: model x=source y, model y=source x.
        sx0, sy0, sz0, sx1, sy1, sz1 = prompt_bbox
        model_prompt_bbox = [sy0, sx0, sz0, sy1, sx1, sz1]
        model_prompt_bbox = self._clamp_bbox(
            model_prompt_bbox, model_shape_xyz
        )
        reconstruction_bbox = self._expand_bbox(
            model_prompt_bbox, int(reconstruction_margin), model_shape_xyz
        )
        print(
            "[MITK reconstruction] "
            f"source_shape_xyz={source_shape_xyz}, model_shape_xyz={model_shape_xyz}, "
            f"MITK_bbox_xyz={prompt_bbox}, model_bbox_xyz={model_prompt_bbox}, "
            f"reconstruction_bbox_xyz={reconstruction_bbox}, checkpoint={self.checkpoint}",
            flush=True,
        )

        xy, xy_rel, xy_box = self._sample_axis_slices(
            volume, model_prompt_bbox, model_prompt_bbox, 0, num_slices)
        xz, xz_rel, xz_box = self._sample_axis_slices(
            volume, model_prompt_bbox, model_prompt_bbox, 1, num_slices)
        yz, yz_rel, yz_box = self._sample_axis_slices(
            volume, model_prompt_bbox, model_prompt_bbox, 2, num_slices)
        print(
            "[MITK reconstruction] "
            f"xy_z={xy_rel.tolist()}, xz_y={xz_rel.tolist()}, yz_x={yz_rel.tolist()}",
            flush=True,
        )
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
        queries, axes = self._make_query_grid(reconstruction_bbox, query_budget)
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
        x0, y0, z0, x1, y1, z1 = reconstruction_bbox
        xi = np.arange(x0, x1 + 1, dtype=np.float32)
        yi = np.arange(y0, y1 + 1, dtype=np.float32)
        zi = np.arange(z0, z1 + 1, dtype=np.float32)
        interpolator = RegularGridInterpolator(
            (zs, ys, xs), coarse, method="linear", bounds_error=False, fill_value=1.0
        )
        zz, yy, xx = np.meshgrid(zi, yi, xi, indexing="ij")
        dense = interpolator(np.stack([zz, yy, xx], axis=-1)).astype(np.float32)
        boundary_negative_ratio = (
            float(np.mean(dense[:, :, 0] < level)),
            float(np.mean(dense[:, :, -1] < level)),
            float(np.mean(dense[:, 0, :] < level)),
            float(np.mean(dense[:, -1, :] < level)),
            float(np.mean(dense[0, :, :] < level)),
            float(np.mean(dense[-1, :, :] < level)),
        )
        verts_zyx, faces, normals, _ = marching_cubes(dense, level=level)
        verts_zyx += np.array([z0, y0, x0], dtype=np.float32)
        verts_model_xyz = verts_zyx[:, [2, 1, 0]]
        # Undo the historical model x/y swap before the MITK client applies
        # the selected image's original IndexToWorld transform.
        verts_xyz = verts_model_xyz[:, [1, 0, 2]]
        # Return raw CT index coordinates. PLY has no standardized medical
        # RAS/LPS metadata, so the MITK client applies its own ImageGeometry
        # IndexToWorld transform after loading the PLY.
        mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)
        mesh.remove_unreferenced_vertices()
        mesh_min = mesh.vertices.min(axis=0)
        mesh_max = mesh.vertices.max(axis=0)
        mesh_bbox = tuple(np.concatenate([mesh_min, mesh_max]).astype(float))
        return ReconstructionResult(
            mesh, sdf_min, sdf_max, len(queries), dense.shape,
            tuple(prompt_bbox), tuple(model_prompt_bbox),
            tuple(reconstruction_bbox), mesh_bbox,
            boundary_negative_ratio,
        )
