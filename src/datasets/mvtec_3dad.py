"""
MVTec3D-AD lightweight Dataset + GPU batched PointMAE/CFM-style preprocessing.

Purpose
-------
This module moves expensive PointMAE preprocessing out of Dataset.__getitem__.
The Dataset only performs file I/O, nearest resize of organized XYZ, and optional
RGB/mask/depth loading. FPS, KNN grouping, background-plane removal, PointMAE
center-relative grouping, and pixel-to-group interpolation maps are computed on
an already-collated batch, preferably on GPU.

Typical usage
-------------
    dataset = MVTec3DAD_PointMAE_Light(
        data_root="/path/to/mvtec_3d",
        category="bagel",
        split="train",
        input_res=224,
        return_rgb=False,
        return_depth=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    cfg = PointMAEGPUPreprocessConfig(num_group=1024, group_size=32, interp_k=3)

    for batch in loader:
        pointmae = pointmae_cfm_preprocess_batch(
            batch["xyz"],
            cfg=cfg,
            device="cuda",
            non_blocking=True,
        )
        groups = pointmae["groups"]       # [B, 1024, 32, 3]
        centers = pointmae["centers"]     # [B, 1024, 3]

Notes
-----
- The default setting is CFM-style high-density grouping with PointMAE-compatible
  local group cardinality: num_group=1024, group_size=32.
- For a stricter CFM public-code-like ablation, set group_size=128.
- KNN is always performed in normalized 3D Euclidean coordinates, not in pixel
  coordinates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import tifffile
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


MVTec3D_CLASSES = [
    "bagel",
    "cable_gland",
    "carrot",
    "cookie",
    "dowel",
    "foam",
    "peach",
    "potato",
    "rope",
    "tire",
]


def _pil_resample(name: str) -> int:
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def imagenet_rgb_tensor_from_pil(
    img: Image.Image,
    size: int,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Small torchvision-free replacement for Resize + ToTensor + Normalize."""
    img = img.resize((size, size), resample=_pil_resample("BICUBIC"))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def binary_mask_tensor_from_pil(img: Image.Image, size: int) -> torch.Tensor:
    """Small torchvision-free replacement for nearest Resize + ToTensor + threshold."""
    img = img.resize((size, size), resample=_pil_resample("NEAREST"))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[..., 0]
    tensor = torch.from_numpy(arr).unsqueeze(0).contiguous()
    return torch.where(tensor > 0.5, 1.0, 0.0)


class MVTec3DAD_PointMAE_Light(Dataset):
    """
    Lightweight MVTec 3D-AD Dataset for PointMAE preprocessing on GPU.

    This class intentionally does NOT run FPS, KNN, RANSAC, or interpolation-map
    construction in __getitem__. Those steps are performed by
    pointmae_cfm_preprocess_batch after DataLoader collation.

    Returned keys:
        xyz:      [3, input_res, input_res], float32 organized point cloud.
        img:      optional [3, rgb_size, rgb_size], ImageNet-normalized RGB.
        depth:    optional [3, input_res, input_res], repeated z channel.
        mask:     optional [1, rgb_size, rgb_size] for test split.
        label:    0 for good, 1 for anomaly.
        metadata: clsname, clslabel, anom_type, paths.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_root: str,
        category: str,
        input_res: int = 224,
        split: str = "train",
        rgb_size: int = 224,
        return_rgb: bool = False,
        return_depth: bool = False,
        is_mask: bool = False,
        rgb_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        return_raw_xyz: bool = False,
        **_: Any,
    ) -> None:
        self.data_root = data_root
        self.category = category
        self.input_res = int(input_res)
        self.rgb_size = int(rgb_size)
        self.split = split
        self.return_rgb = return_rgb
        self.return_depth = return_depth
        self.is_mask = is_mask
        self.return_raw_xyz = return_raw_xyz

        assert Path(data_root).exists(), f"Path {data_root} does not exist"
        assert category in MVTec3D_CLASSES, f"Unknown category: {category}"
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        if rgb_transform is None:
            self.rgb_transform = lambda img: imagenet_rgb_tensor_from_pil(
                img,
                size=self.rgb_size,
                mean=tuple(self.IMAGENET_MEAN),
                std=tuple(self.IMAGENET_STD),
            )
        else:
            self.rgb_transform = rgb_transform

        if mask_transform is None:
            self.mask_transform = lambda img: binary_mask_tensor_from_pil(img, size=self.rgb_size)
        else:
            self.mask_transform = mask_transform

        self.img_files = self.get_files()
        self.labels = []
        for img_file in self.img_files:
            anom_type = img_file.parent.parent.name
            self.labels.append(0 if anom_type == "good" else 1)

        self.num_classes = len(MVTec3D_CLASSES)
        self.labels_to_names = {i: name for i, name in enumerate(MVTec3D_CLASSES)}

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_file = self.img_files[index]
        xyz_file = self.get_xyz_file(img_file)

        cls_name = self.category
        cls_label = MVTec3D_CLASSES.index(cls_name)
        anom_type = img_file.parent.parent.name
        label = self.labels[index]

        organized_pc = self.read_tiff_organized_pc(xyz_file)
        xyz = self.preprocess_organized_pc(organized_pc)

        sample: Dict[str, Any] = {
            "xyz": xyz,  # [3, input_res, input_res]
            "label": int(label),
            "clsname": cls_name,
            "clslabel": int(cls_label),
            "anom_type": anom_type,
            "path": str(img_file),
            "rgb_path": str(img_file),
            "xyz_path": str(xyz_file),
        }

        if self.return_rgb:
            img_pil = Image.open(img_file).convert("RGB")
            img = self.rgb_transform(img_pil)
            sample["img"] = img
            sample.setdefault("feature", {})["rgb"] = img

        if self.return_depth:
            depth = xyz[2:3].repeat(3, 1, 1).contiguous()
            sample["depth"] = depth
            sample.setdefault("feature", {})["depth"] = depth

        if self.split == "test" and self.is_mask:
            if anom_type == "good":
                mask = torch.zeros((1, self.rgb_size, self.rgb_size), dtype=torch.float32)
            else:
                mask_file = self.get_mask_file(img_file)
                mask_pil = Image.open(mask_file).convert("L")
                mask = self.mask_transform(mask_pil)[:1]
            sample["mask"] = mask

        if self.return_raw_xyz:
            sample["raw_xyz"] = torch.from_numpy(organized_pc).permute(2, 0, 1).float().contiguous()

        return sample

    def get_files(self):
        if self.split == "train":
            return sorted(
                Path(os.path.join(self.data_root, self.category, "train", "good", "rgb")).glob("*.png")
            )

        test_root = Path(os.path.join(self.data_root, self.category, "test"))
        normal_files = sorted((test_root / "good" / "rgb").glob("*.png"))

        anomaly_files = []
        for defect_dir in sorted(test_root.glob("*")):
            if defect_dir.name == "good":
                continue
            anomaly_files += sorted((defect_dir / "rgb").glob("*.png"))

        return normal_files + sorted(anomaly_files)

    def get_xyz_file(self, img_file: Path) -> Path:
        xyz_file = img_file.parent.parent / "xyz" / img_file.name.replace(".png", ".tiff")
        if not xyz_file.exists():
            xyz_file = img_file.parent.parent / "xyz" / img_file.name.replace(".png", ".tif")
        assert xyz_file.exists(), f"XYZ file not found: {xyz_file}"
        return xyz_file

    def get_mask_file(self, img_file: Path) -> Path:
        mask_file = img_file.parent.parent / "gt" / img_file.name
        assert mask_file.exists(), f"Mask file not found: {mask_file}"
        return mask_file

    @staticmethod
    def read_tiff_organized_pc(xyz_file: Path) -> np.ndarray:
        organized_pc = tifffile.imread(str(xyz_file)).astype(np.float32)
        if organized_pc.ndim == 2:
            organized_pc = organized_pc[..., None]
        if organized_pc.shape[-1] == 1:
            organized_pc = np.repeat(organized_pc, 3, axis=-1)
        assert organized_pc.ndim == 3, f"Invalid xyz shape: {organized_pc.shape}"
        assert organized_pc.shape[-1] >= 3, f"Expected xyz with 3 channels, got {organized_pc.shape}"
        return organized_pc[..., :3]

    def preprocess_organized_pc(self, organized_pc: np.ndarray) -> torch.Tensor:
        """Nearest resize only. This is kept in Dataset so default collate can batch tensors."""
        pc = torch.from_numpy(organized_pc).float().permute(2, 0, 1).unsqueeze(0).contiguous()
        if pc.shape[-2:] != (self.input_res, self.input_res):
            pc = F.interpolate(pc, size=(self.input_res, self.input_res), mode="nearest")
        return pc.squeeze(0).contiguous().float()


@dataclass
class PointMAEGPUPreprocessConfig:
    """Configuration for batched PointMAE/CFM-style preprocessing."""

    num_group: int = 1024
    group_size: int = 32
    interp_k: int = 3

    # Background plane removal, following the CFM/M3DM preprocessing intent.
    remove_background_plane: bool = True
    plane_edge_width: int = 10
    plane_ransac_threshold: float = 0.004
    plane_remove_threshold: float = 0.005
    plane_ransac_iters: int = 128
    plane_refine: bool = True

    # PointMAE-style per-sample coordinate normalization.
    normalize: str = "unit_sphere"  # 'unit_sphere' or 'none'
    eps: float = 1e-8

    # Chunking to control memory.
    knn_chunk_size: int = 128
    interp_chunk_size: int = 8192

    # Robustness for samples with too few foreground points.
    fallback_to_valid_if_few_points: bool = True
    min_foreground_points: int = 128

    # FPS initialization.
    fps_start: str = "centroid_farthest"  # 'centroid_farthest' or 'random'

    # Optional determinism for GPU RANSAC/FPS random start.
    deterministic: bool = False
    random_seed: int = 0


def _as_device(device: Optional[Union[str, torch.device]], fallback: torch.device) -> torch.device:
    if device is None:
        return fallback
    return torch.device(device)


def _flatten_xyz(xyz: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if xyz.dim() != 4 or xyz.size(1) != 3:
        raise ValueError(f"xyz must have shape [B, 3, H, W], got {tuple(xyz.shape)}")
    b, _, h, w = xyz.shape
    points = xyz.permute(0, 2, 3, 1).reshape(b, h * w, 3).contiguous()
    return points, h, w


def make_valid_xyz_mask_torch(xyz: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Args:
        xyz: [B, 3, H, W]
    Returns:
        valid_mask: [B, H, W], bool.
    """
    finite = torch.isfinite(xyz).all(dim=1)
    norm = torch.linalg.norm(torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    return finite & (norm > eps)


def _edge_mask_like(valid_mask: torch.Tensor, edge_width: int) -> torch.Tensor:
    b, h, w = valid_mask.shape
    ew = max(1, min(int(edge_width), h // 2, w // 2))
    edge = torch.zeros_like(valid_mask, dtype=torch.bool)
    edge[:, :ew, :] = True
    edge[:, -ew:, :] = True
    edge[:, :, :ew] = True
    edge[:, :, -ew:] = True
    return edge & valid_mask


def _normalize_plane(plane: torch.Tensor, eps: float) -> torch.Tensor:
    normal = plane[..., :3]
    denom = torch.linalg.norm(normal, dim=-1, keepdim=True).clamp_min(eps)
    return torch.cat([normal / denom, plane[..., 3:4] / denom], dim=-1)


def fit_background_planes_ransac_gpu(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: PointMAEGPUPreprocessConfig,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched GPU RANSAC plane fitting from edge points.

    Args:
        xyz: [B, 3, H, W]
        valid_mask: [B, H, W]
    Returns:
        planes: [B, 4], normalized ax+by+cz+d=0.
        has_plane: [B], bool.
    """
    device = xyz.device
    dtype = xyz.dtype
    b, _, h, w = xyz.shape
    edge_mask = _edge_mask_like(valid_mask, cfg.plane_edge_width).reshape(b, h * w)
    points_flat, _, _ = _flatten_xyz(torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0))

    planes = torch.zeros((b, 4), device=device, dtype=dtype)
    has_plane = torch.zeros((b,), device=device, dtype=torch.bool)

    for bi in range(b):
        pts = points_flat[bi, edge_mask[bi]]
        if pts.size(0) < 3:
            continue

        num_pts = pts.size(0)
        num_iters = max(1, int(cfg.plane_ransac_iters))
        sample_idx = torch.randint(
            low=0,
            high=num_pts,
            size=(num_iters, 3),
            device=device,
            generator=generator,
        )
        tri = pts[sample_idx]  # [I, 3, 3]
        p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
        normals = torch.cross(p1 - p0, p2 - p0, dim=-1)
        normal_norm = torch.linalg.norm(normals, dim=-1)
        valid_planes = normal_norm > cfg.eps
        normals = normals / normal_norm.clamp_min(cfg.eps).unsqueeze(-1)
        d = -(normals * p0).sum(dim=-1)

        # distances: [num_pts, num_iters]
        distances = torch.abs(pts @ normals.transpose(0, 1) + d.unsqueeze(0))
        inlier_counts = (distances < cfg.plane_ransac_threshold).sum(dim=0)
        inlier_counts = torch.where(
            valid_planes,
            inlier_counts,
            torch.full_like(inlier_counts, -1),
        )
        best_idx = torch.argmax(inlier_counts)
        if inlier_counts[best_idx] < 3:
            continue

        best_normal = normals[best_idx]
        best_d = d[best_idx]
        best_plane = torch.cat([best_normal, best_d.view(1)], dim=0)

        if cfg.plane_refine:
            best_dist = torch.abs(pts @ best_normal + best_d)
            inliers = pts[best_dist < cfg.plane_ransac_threshold]
            if inliers.size(0) >= 3:
                centroid = inliers.mean(dim=0, keepdim=True)
                centered = inliers - centroid
                try:
                    # Vh[-1] is the least-variance direction, i.e. plane normal.
                    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                    refined_normal = vh[-1]
                    refined_normal = refined_normal / torch.linalg.norm(refined_normal).clamp_min(cfg.eps)
                    refined_d = -(refined_normal * centroid.squeeze(0)).sum()
                    best_plane = torch.cat([refined_normal, refined_d.view(1)], dim=0)
                except RuntimeError:
                    pass

        planes[bi] = _normalize_plane(best_plane, cfg.eps)
        has_plane[bi] = True

    return planes, has_plane


def foreground_mask_cfm_gpu(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: PointMAEGPUPreprocessConfig,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute CFM/M3DM-style foreground mask on GPU.

    Returns:
        foreground_mask: [B, H, W], bool.
        planes: [B, 4].
        has_plane: [B], bool.
    """
    if not cfg.remove_background_plane:
        b = xyz.size(0)
        planes = torch.zeros((b, 4), device=xyz.device, dtype=xyz.dtype)
        has_plane = torch.zeros((b,), device=xyz.device, dtype=torch.bool)
        return valid_mask, planes, has_plane

    planes, has_plane = fit_background_planes_ransac_gpu(xyz, valid_mask, cfg, generator=generator)
    points_flat, h, w = _flatten_xyz(torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0))
    valid_flat = valid_mask.reshape(xyz.size(0), h * w)

    normal = planes[:, :3]
    d = planes[:, 3]
    dist = torch.abs((points_flat * normal[:, None, :]).sum(dim=-1) + d[:, None])
    fg_flat = valid_flat & (dist >= cfg.plane_remove_threshold) & has_plane[:, None]

    # If no reliable plane was found, use all valid points.
    fg_flat = torch.where(has_plane[:, None], fg_flat, valid_flat)

    if cfg.fallback_to_valid_if_few_points:
        fg_count = fg_flat.sum(dim=1)
        valid_count = valid_flat.sum(dim=1)
        use_valid = (fg_count < cfg.min_foreground_points) & (valid_count > 0)
        fg_flat = torch.where(use_valid[:, None], valid_flat, fg_flat)

    return fg_flat.reshape(xyz.size(0), h, w), planes, has_plane


def normalize_points_masked(
    points: torch.Tensor,
    mask: torch.Tensor,
    cfg: PointMAEGPUPreprocessConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PointMAE/ShapeNet-style per-sample unit-sphere normalization.

    Args:
        points: [B, N, 3]
        mask: [B, N], bool, points used to estimate center and scale.
    Returns:
        points_norm: [B, N, 3]
        center: [B, 3]
        scale: [B, 1]
    """
    points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    if cfg.normalize == "none":
        center = torch.zeros((points.size(0), 3), device=points.device, dtype=points.dtype)
        scale = torch.ones((points.size(0), 1), device=points.device, dtype=points.dtype)
        return points, center, scale
    if cfg.normalize != "unit_sphere":
        raise ValueError(f"Unsupported normalization mode: {cfg.normalize}")

    mask_f = mask.to(dtype=points.dtype)
    counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    center = (points * mask_f.unsqueeze(-1)).sum(dim=1) / counts
    centered = points - center.unsqueeze(1)
    radius = torch.linalg.norm(centered, dim=-1)
    radius = radius.masked_fill(~mask, 0.0)
    scale = radius.max(dim=1, keepdim=True).values.clamp_min(cfg.eps)
    points_norm = centered / scale.unsqueeze(-1)
    points_norm = torch.nan_to_num(points_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return points_norm, center, scale


def batch_gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Batched gather for point tensors.

    Args:
        points: [B, N, C]
        idx: [B, ...]
    Returns:
        gathered: [B, ..., C]
    """
    if points.dim() != 3:
        raise ValueError(f"points must be [B,N,C], got {tuple(points.shape)}")
    b, _, c = points.shape
    if idx.size(0) != b:
        raise ValueError("Batch dimension mismatch between points and idx")
    idx_flat = idx.reshape(b, -1).long()
    gathered = points.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, c))
    return gathered.reshape(*idx.shape, c)


def farthest_point_sampling_batched(
    points: torch.Tensor,
    valid_mask: torch.Tensor,
    num_samples: int,
    start: str = "centroid_farthest",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Batched FPS over masked point clouds.

    Args:
        points: [B, N, 3]
        valid_mask: [B, N], bool.
        num_samples: number of centers.
    Returns:
        fps_idx: [B, num_samples]
    """
    b, n, _ = points.shape
    device = points.device
    fps_idx = torch.zeros((b, num_samples), device=device, dtype=torch.long)

    # Guarantee at least one selectable point per sample.
    has_valid = valid_mask.any(dim=1)
    if not bool(has_valid.all()):
        valid_mask = valid_mask.clone()
        valid_mask[~has_valid, 0] = True

    mask_f = valid_mask.to(dtype=points.dtype)
    counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    centroid = (points * mask_f.unsqueeze(-1)).sum(dim=1) / counts

    if start == "random":
        rand_scores = torch.rand((b, n), device=device, generator=generator)
        rand_scores = rand_scores.masked_fill(~valid_mask, -1.0)
        farthest = torch.argmax(rand_scores, dim=1)
    elif start == "centroid_farthest":
        dist_to_centroid = ((points - centroid[:, None, :]) ** 2).sum(dim=-1)
        dist_to_centroid = dist_to_centroid.masked_fill(~valid_mask, -1.0)
        farthest = torch.argmax(dist_to_centroid, dim=1)
    else:
        raise ValueError(f"Unsupported FPS start mode: {start}")

    min_dist = torch.full((b, n), float("inf"), device=device, dtype=points.dtype)
    batch_arange = torch.arange(b, device=device)

    for i in range(num_samples):
        fps_idx[:, i] = farthest
        center = points[batch_arange, farthest].unsqueeze(1)  # [B,1,3]
        dist = ((points - center) ** 2).sum(dim=-1)
        min_dist = torch.minimum(min_dist, dist)
        min_dist = min_dist.masked_fill(~valid_mask, -1.0)
        farthest = torch.argmax(min_dist, dim=1)

    return fps_idx


def knn_indices_batched_chunked(
    queries: torch.Tensor,
    refs: torch.Tensor,
    k: int,
    ref_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Chunked batched KNN in Euclidean 3D space.

    Args:
        queries: [B, Q, 3]
        refs: [B, N, 3]
        k: number of nearest neighbors.
        ref_mask: optional [B, N] bool mask for valid references.
    Returns:
        idx: [B, Q, k]
    """
    b, q, _ = queries.shape
    _, n, _ = refs.shape
    if n == 0:
        raise ValueError("KNN received an empty reference tensor")
    k_eff = min(int(k), n)
    out = torch.empty((b, q, k_eff), device=queries.device, dtype=torch.long)

    if ref_mask is None:
        ref_mask = torch.ones((b, n), device=queries.device, dtype=torch.bool)
    else:
        ref_mask = ref_mask.bool()
        has_ref = ref_mask.any(dim=1)
        if not bool(has_ref.all()):
            ref_mask = ref_mask.clone()
            ref_mask[~has_ref, 0] = True

    first_valid = torch.argmax(ref_mask.to(torch.long), dim=1)

    for start in range(0, q, int(chunk_size)):
        end = min(start + int(chunk_size), q)
        q_chunk = queries[:, start:end, :]  # [B, Cq, 3]
        dist = ((q_chunk[:, :, None, :] - refs[:, None, :, :]) ** 2).sum(dim=-1)
        dist = dist.masked_fill(~ref_mask[:, None, :], float("inf"))
        _, idx = torch.topk(dist, k=k_eff, dim=-1, largest=False, sorted=True)

        gathered_valid = ref_mask.gather(1, idx.reshape(b, -1)).reshape(b, end - start, k_eff)
        idx = torch.where(gathered_valid, idx, first_valid[:, None, None].expand_as(idx))
        out[:, start:end, :] = idx

    if k_eff < k:
        repeat = int(np.ceil(k / k_eff))
        out = out.repeat(1, 1, repeat)[:, :, :k]

    return out


def inverse_distance_interpolation_weights(
    query_points: torch.Tensor,
    center_points: torch.Tensor,
    neighbor_idx: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args:
        query_points: [B, Q, 3]
        center_points: [B, G, 3]
        neighbor_idx: [B, Q, K]
        query_mask: optional [B, Q], zeroes out invalid/background queries.
    Returns:
        weights: [B, Q, K]
    """
    neighbor_centers = batch_gather_points(center_points, neighbor_idx)  # [B,Q,K,3]
    dist = torch.linalg.norm(query_points.unsqueeze(2) - neighbor_centers, dim=-1)
    inv = 1.0 / (dist + eps)
    weights = inv / inv.sum(dim=-1, keepdim=True).clamp_min(eps)
    if query_mask is not None:
        weights = weights * query_mask.to(dtype=weights.dtype).unsqueeze(-1)
    return weights


def pointmae_cfm_preprocess_batch(
    xyz: Union[torch.Tensor, Dict[str, Any]],
    cfg: Optional[PointMAEGPUPreprocessConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = True,
    return_pixel_maps: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    GPU batched PointMAE/CFM-style preprocessing.

    Args:
        xyz:
            Either a tensor [B,3,H,W] or a DataLoader batch dict containing key
            "xyz". The tensor is moved to device inside this function.
        cfg:
            Preprocessing config. Defaults to PointMAEGPUPreprocessConfig().
        device:
            Target device. If None, uses xyz.device. For GPU preprocessing, pass
            "cuda" or torch.device("cuda").
        non_blocking:
            Used when moving CPU tensors to GPU. Use DataLoader(pin_memory=True).
        return_pixel_maps:
            If False, skips pixel-to-group interpolation maps. This is faster and
            useful if you only need PointMAE tokens during training.

    Returns:
        Dict with:
            groups: [B, G, S, 3], center-relative PointMAE local groups.
            centers: [B, G, 3], normalized group centers.
            center_pixels: [B, G, 2], row/col of sampled centers.
            group_knn_idx: [B, G, S], flat pixel indices used in each group.
            valid_mask: [B,1,H,W].
            foreground_mask: [B,1,H,W].
            normalization_center: [B,3].
            normalization_scale: [B,1].
            background_plane: [B,4].
            has_background_plane: [B].
            num_foreground_points: [B].
            pixel_to_group_idx/weight if return_pixel_maps=True.
    """
    if cfg is None:
        cfg = PointMAEGPUPreprocessConfig()

    if isinstance(xyz, dict):
        xyz_tensor = xyz["xyz"]
    else:
        xyz_tensor = xyz

    target_device = _as_device(device, xyz_tensor.device)
    xyz_tensor = xyz_tensor.to(device=target_device, dtype=torch.float32, non_blocking=non_blocking).contiguous()
    if xyz_tensor.dim() != 4 or xyz_tensor.size(1) != 3:
        raise ValueError(f"xyz must be [B,3,H,W], got {tuple(xyz_tensor.shape)}")

    generator: Optional[torch.Generator] = None
    if cfg.deterministic:
        generator = torch.Generator(device=target_device)
        generator.manual_seed(int(cfg.random_seed))

    b, _, h, w = xyz_tensor.shape
    xyz_clean = torch.nan_to_num(xyz_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = make_valid_xyz_mask_torch(xyz_tensor, eps=cfg.eps)  # [B,H,W]

    # If a sample has no valid points, make one dummy valid point to avoid crashes.
    valid_flat = valid_mask.reshape(b, h * w)
    no_valid = ~valid_flat.any(dim=1)
    if bool(no_valid.any()):
        valid_flat = valid_flat.clone()
        valid_flat[no_valid, 0] = True
        valid_mask = valid_flat.reshape(b, h, w)

    foreground_mask, planes, has_plane = foreground_mask_cfm_gpu(
        xyz_clean,
        valid_mask,
        cfg,
        generator=generator,
    )
    fg_flat = foreground_mask.reshape(b, h * w)

    points, _, _ = _flatten_xyz(xyz_clean)  # [B,N,3]
    points_norm, norm_center, norm_scale = normalize_points_masked(points, fg_flat, cfg)

    center_idx = farthest_point_sampling_batched(
        points_norm,
        fg_flat,
        num_samples=cfg.num_group,
        start=cfg.fps_start,
        generator=generator,
    )  # [B,G]
    centers = batch_gather_points(points_norm, center_idx)  # [B,G,3]

    group_knn_idx = knn_indices_batched_chunked(
        queries=centers,
        refs=points_norm,
        k=cfg.group_size,
        ref_mask=fg_flat,
        chunk_size=cfg.knn_chunk_size,
    )  # [B,G,S]
    neighborhoods_abs = batch_gather_points(points_norm, group_knn_idx)  # [B,G,S,3]
    groups = neighborhoods_abs - centers.unsqueeze(2)

    center_rows = torch.div(center_idx, w, rounding_mode="floor")
    center_cols = center_idx % w
    center_pixels = torch.stack([center_rows, center_cols], dim=-1).long()

    output: Dict[str, torch.Tensor] = {
        "groups": groups.contiguous(),
        "centers": centers.contiguous(),
        "center_indices": center_idx.long().contiguous(),
        "center_pixels": center_pixels.contiguous(),
        "group_knn_idx": group_knn_idx.long().contiguous(),
        "valid_mask": valid_mask[:, None].to(dtype=torch.float32).contiguous(),
        "foreground_mask": foreground_mask[:, None].to(dtype=torch.float32).contiguous(),
        "normalization_center": norm_center.contiguous(),
        "normalization_scale": norm_scale.contiguous(),
        "background_plane": planes.contiguous(),
        "has_background_plane": has_plane.long().contiguous(),
        "num_foreground_points": fg_flat.sum(dim=1).long().contiguous(),
    }

    if return_pixel_maps:
        pixel_to_group_idx = knn_indices_batched_chunked(
            queries=points_norm,
            refs=centers,
            k=cfg.interp_k,
            ref_mask=None,
            chunk_size=cfg.interp_chunk_size,
        )  # [B,N,K]
        pixel_to_group_weight = inverse_distance_interpolation_weights(
            query_points=points_norm,
            center_points=centers,
            neighbor_idx=pixel_to_group_idx,
            query_mask=fg_flat,
            eps=cfg.eps,
        )
        output["pixel_to_group_idx"] = pixel_to_group_idx.reshape(b, h, w, cfg.interp_k).long().contiguous()
        output["pixel_to_group_weight"] = pixel_to_group_weight.reshape(b, h, w, cfg.interp_k).contiguous()

    return output


def interpolate_group_values_to_grid_gpu(
    group_values: torch.Tensor,
    pixel_to_group_idx: torch.Tensor,
    pixel_to_group_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate group-level scores/features to a dense grid.

    Args:
        group_values:
            [B,G] scalar scores or [B,G,C] features.
        pixel_to_group_idx:
            [B,H,W,K].
        pixel_to_group_weight:
            [B,H,W,K].
    Returns:
        [B,1,H,W] for scalar scores, or [B,C,H,W] for features.
    """
    if group_values.dim() not in (2, 3):
        raise ValueError(f"group_values must be [B,G] or [B,G,C], got {tuple(group_values.shape)}")
    b, h, w, k = pixel_to_group_idx.shape
    if group_values.size(0) != b:
        raise ValueError("Batch dimension mismatch")

    idx = pixel_to_group_idx.to(device=group_values.device, dtype=torch.long)
    weight = pixel_to_group_weight.to(device=group_values.device, dtype=group_values.dtype)
    batch_idx = torch.arange(b, device=group_values.device).view(b, 1, 1, 1).expand(b, h, w, k)

    if group_values.dim() == 2:
        gathered = group_values[batch_idx, idx]  # [B,H,W,K]
        dense = (gathered * weight).sum(dim=-1)
        return dense.unsqueeze(1).contiguous()

    gathered = group_values[batch_idx, idx]  # [B,H,W,K,C]
    dense = (gathered * weight.unsqueeze(-1)).sum(dim=-2)  # [B,H,W,C]
    return dense.permute(0, 3, 1, 2).contiguous()


# Backward-compatible alias for concise imports.
MVTec3DAD_PointMAE_GPU = MVTec3DAD_PointMAE_Light


if __name__ == "__main__":
    # Synthetic smoke test. This does not require the MVTec3D-AD files.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, h, w = 2, 64, 64
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    zz = 0.05 * torch.randn(h, w)
    xyz = torch.stack([xx, yy, zz], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
    xyz[:, :, :4, :] = 0  # fake invalid/background-ish region

    cfg = PointMAEGPUPreprocessConfig(
        num_group=64,
        group_size=16,
        interp_k=3,
        plane_ransac_iters=16,
        knn_chunk_size=32,
        interp_chunk_size=1024,
        deterministic=True,
        random_seed=0,
    )
    out = pointmae_cfm_preprocess_batch(xyz, cfg=cfg, device=device)
    print("groups", tuple(out["groups"].shape))
    print("centers", tuple(out["centers"].shape))
    print("pixel_to_group_idx", tuple(out["pixel_to_group_idx"].shape))
    scores = torch.randn(b, cfg.num_group, device=device)
    dense = interpolate_group_values_to_grid_gpu(scores, out["pixel_to_group_idx"], out["pixel_to_group_weight"])
    print("dense", tuple(dense.shape))