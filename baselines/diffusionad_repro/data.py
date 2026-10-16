"""Multi-class version of DiffusionAD's training set (data/dataset_beta_thresh.py).

The anomaly synthesis (`perlin_synthetic`) and augmenter lists are copied from
MVTecTrainDataset of the upstream repo; the per-dataset train classes differ only
in where the foreground comes from, which is reproduced per sample:
  * texture classes (upstream texture_list; only MVTec has any): a random mask
    from the authors' 32 `carpet/thresh` masks,
  * all other classes: the image's own DIS foreground mask (`DISthresh`).
Multi-class change: one dataset over the union of all classes of a benchmark;
like upstream, `__getitem__` ignores `idx` and draws a uniformly random image
(so classes are sampled in proportion to their train-set size).
Inputs come from prepare_cache.py (uint8 arrays after upstream's cv2.resize).
"""
import os
import random

import imgaug
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset

from .orig.perlin import rand_perlin_2d_np


class DiffusionADTrain(Dataset):

    def __init__(self, cache_dir, img_size=(256, 256)):
        aux = os.path.dirname(cache_dir.rstrip('/'))
        import json
        meta = json.load(open(os.path.join(cache_dir, 'meta.json')))
        self.images = np.load(os.path.join(cache_dir, 'train_img.npy'), mmap_mode='r')
        self.fg = np.load(os.path.join(cache_dir, 'train_fg.npy'), mmap_mode='r')
        self.cls = np.load(os.path.join(cache_dir, 'train_cls.npy'))
        self.is_texture = np.array(meta['texture'], dtype=bool)
        self.anomaly_sources = np.load(os.path.join(aux, 'dtd.npy'), mmap_mode='r')
        self.textural_foreground = np.load(os.path.join(aux, 'texture_fg.npy'))
        self.resize_shape = [img_size[0], img_size[1]]

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.images)

    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def perlin_synthetic(self, image, thresh, anomaly_source_img, cv2_image, is_texture):
        # verbatim from upstream MVTecTrainDataset.perlin_synthetic, except that the
        # DTD image arrives already read + resized and texture-ness is a flag
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt < 50):
                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))

                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(np.float32)

                msk = (object_perlin).astype(np.float32)
                if np.sum(msk) != 0:
                    has_anomaly = 1
                try_cnt += 1

            if is_texture:  # only DTD
                aug = self.randAugmenter()
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else:  # DTD and self-augmentation
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  # >0.5 is DTD
                    aug = self.randAugmenter()
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else:  # self-augmentation
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0

            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2
            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.images), (1,)).item()   # upstream ignores idx
        image = np.ascontiguousarray(self.images[idx])            # uint8 RGB, already resized
        cv2_image = image
        is_texture = bool(self.is_texture[self.cls[idx]])
        if is_texture:   # upstream random_choice_foreground_path()
            thresh = self.textural_foreground[torch.randint(0, len(self.textural_foreground), (1,)).item()]
        else:            # upstream image_path.replace('train', 'DISthresh')
            thresh = self.fg[idx]
        thresh = np.array(thresh).astype(np.float32)/255.0
        image = np.array(image).astype(np.float32)/255.0

        anomaly_source_idx = torch.randint(0, len(self.anomaly_sources), (1,)).item()
        anomaly_source_img = np.ascontiguousarray(self.anomaly_sources[anomaly_source_idx])
        augmented_image, anomaly_mask, has_anomaly = self.perlin_synthetic(
            image, thresh, anomaly_source_img, cv2_image, is_texture)

        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        return {'image': image, 'anomaly_mask': anomaly_mask, 'augmented_image': augmented_image,
                'has_anomaly': has_anomaly, 'idx': idx, 'cls': int(self.cls[idx])}


def worker_init_fn(worker_id):
    """Not upstream: give each worker (and each DDP rank) its own imgaug and
    python-random stream. torch / numpy are already seeded per worker by PyTorch
    from the (per-rank) base seed; imgaug's global RNG and `random` would
    otherwise be identical copies in every forked worker."""
    s = torch.initial_seed() % (2 ** 32)
    imgaug.seed(s)
    random.seed(s)
