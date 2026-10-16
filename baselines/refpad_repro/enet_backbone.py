"""EfficientNet-B4 token backbone for the ReFP-AD reproduction.

Drop-in replacement for ReFP-AD's `backbone.DinoBackbone` (same constructor
signature, same `extract` / `extract_batch` -> (N_patches, out_dim) contract,
same `gh/gw/num_patches/out_dim` attributes), so the unmodified ReFP-AD
pipeline / TokenCache / _score_image code runs against it unchanged.

Feature definition follows FlowAD exactly (configs/train/ad_ditB_enet_edm.yaml
`model.backbone`): efficientnet-b4 via src.backbones.get_backbone with
outblocks (1, 5, 9, 21) at strides (2, 4, 8, 16), each map bilinearly rescaled
to 16x16 and channel-concatenated -> (272, 16, 16); input is FlowAD's
`imagenet` transform at 256x256 (plain Resize, no crop). Tokens are the raw
concatenated features (ReFP-AD applies its own per-category TokenStats
z-scoring downstream, as in its original code).
"""

from pathlib import Path
from typing import Sequence, Tuple, Union

import torch
import torchvision.transforms as T
from PIL import Image

PathLike = Union[str, Path]

IMG_SIZE = 256
FEAT_C, FEAT_H, FEAT_W = 272, 16, 16  # get_backbone_feature_shape("efficientnet-b4")


class EnetBackbone:
    """FlowAD efficientnet-b4 multi-scale features as ReFP-AD patch tokens."""

    def __init__(
        self,
        model_name: str,
        device: str,
        resize: Tuple[int, int],
        center_crop: Tuple[int, int],
        patch=None,
        aggregation=None,
        layers=None,
        dtype=None,
    ):
        from src.backbones import get_backbone, get_backbone_feature_shape

        if tuple(resize) != (IMG_SIZE, IMG_SIZE) or tuple(center_crop) != (IMG_SIZE, IMG_SIZE):
            raise ValueError(
                f"EnetBackbone expects resize=center_crop=({IMG_SIZE},{IMG_SIZE}) "
                f"(FlowAD protocol), got resize={resize} center_crop={center_crop}"
            )

        self.model_name = "efficientnet-b4"
        self.device = torch.device(device)
        self.resize = tuple(resize)
        self.center_crop = tuple(center_crop)
        self.aggregation = aggregation
        self.layers = tuple(layers) if layers is not None else ()

        # FlowAD's efficientnet feature_res default changed 16 -> 32; this repro (and the
        # RealIAD run) uses the 16x16 grid, so pin it
        c, h, w = get_backbone_feature_shape("efficientnet-b4", feature_res=FEAT_H)
        assert (c, h, w) == (FEAT_C, FEAT_H, FEAT_W)
        self.gh, self.gw = h, w
        self.num_patches = h * w
        self.patch = IMG_SIZE // h  # 16

        # identical kwargs to FlowAD's model.backbone section
        self.model = get_backbone(
            model_name="efficientnet-b4",
            pretrained=True,
            outblocks=[1, 5, 9, 21],
            outstrides=[2, 4, 8, 16],
            feature_res=FEAT_H,
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # FlowAD build_transforms(img_size=256, transform_type="imagenet")
        self.transform = T.Compose(
            [
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        print(
            f"[-] Loading backbone: efficientnet-b4 (FlowAD impl) | "
            f"dim={FEAT_C}, grid={self.gh}x{self.gw}, input={IMG_SIZE}"
        )

    @property
    def out_dim(self) -> int:
        return FEAT_C

    @torch.inference_mode()
    def extract_batch(self, image_paths: Sequence[PathLike]) -> torch.Tensor:
        """Returns CPU float32 tensor of shape (B, N_patches, out_dim)."""
        if not image_paths:
            raise ValueError("image_paths cannot be empty.")
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(self.transform(image.convert("RGB")))
        x = torch.stack(images).to(self.device)
        feats, _ = self.model(x)  # (B, 272, 16, 16)
        toks = feats.flatten(2).permute(0, 2, 1).contiguous()  # (B, 256, 272)
        return toks.float().cpu()

    @torch.inference_mode()
    def extract(self, img_path: PathLike) -> torch.Tensor:
        """Returns CPU float32 tensor of shape (N_patches, out_dim)."""
        return self.extract_batch([img_path])[0]
