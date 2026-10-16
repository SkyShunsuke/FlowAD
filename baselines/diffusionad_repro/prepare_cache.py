"""Build the 256x256 uint8 caches used by the DiffusionAD reproduction.

DiffusionAD's data pipeline (data/dataset_beta_thresh.py) starts every sample
with `cv2.resize(x, (256, 256))` on uint8 arrays: the train/test image, the
foreground mask and the DTD anomaly source. Doing that resize once here and
storing the uint8 result is bit-exact and removes JPEG decoding of large
images (Real-IAD, VisA) from the training loop.

Foreground masks (`DISthresh`) are regenerated with DIS / IS-Net exactly as
the authors did: isnet-general-use.pth, DIS/IS-Net/Inference.py preprocessing,
binarised at 0.5. tools/calib_dis.py checks this against the authors' released
masks (IoU 1.000 on bottle / pill / candle / pcb1). The authors only released
masks for MVTec-AD and VisA; MPDD and Real-IAD get masks from the same
pipeline. Texture classes use the authors' 32 random `carpet/thresh` masks.

Splits, class order and test GT come from FlowAD's src/datasets (same as the
other baselines); test masks are FlowAD's 256 NEAREST-resized masks.

Output: data/diffusionad_aux/cache/<dataset>/
  train_img.npy (N,256,256,3) RGB, train_fg.npy (N,256,256), train_cls.npy (N,)
  test_img.npy (M,256,256,3) RGB, test_mask.npy (M,256,256) {0,1},
  test_label.npy (M,), test_cls.npy (M,), meta.json
and data/diffusionad_aux/cache/{dtd.npy, texture_fg.npy}.

Run (from the repo root):
  torchrun --standalone --nproc-per-node=8 baselines/diffusionad_repro/prepare_cache.py --datasets mvtec mpdd visa realiad
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torchvision.transforms.functional import normalize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.datasets import build_dataset  # noqa: E402

AUX = os.path.join(ROOT, 'data', 'diffusionad_aux')
CACHE = os.environ.get('DIFFAD_CACHE', os.path.join(AUX, 'cache'))   # Real-IAD: node-local disk
sys.path.insert(0, os.path.join(AUX, 'DIS', 'IS-Net'))

# texture_list of dataset_beta_thresh.py (zipper is treated as a texture there)
TEXTURES = {'carpet', 'zipper', 'leather', 'tile', 'wood', 'grid'}
DATASETS = {
    'mvtec': dict(dataset_name='mvtec_ad_all', data_root='data/mvtec_ad'),
    'visa': dict(dataset_name='visa_all', data_root='data/visa'),
    'mpdd': dict(dataset_name='mpdd_all', data_root='data/mpdd'),
    'realiad': dict(dataset_name='realiad_all', data_root='data/realiad',
                    meta_dir='./data/realiad_meta/realiad_jsons'),
}
S = 256


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def resize(x):
    return cv2.resize(x, dsize=(S, S))   # as in dataset_beta_thresh.py (INTER_LINEAR)


class Items(torch.utils.data.Dataset):
    """Loads full-res RGB for DIS (if `dis`) plus the 256 resized image."""

    def __init__(self, paths, dis):
        self.paths, self.dis = paths, dis

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = read_rgb(self.paths[i])
        out = {'i': i, 'small': torch.from_numpy(resize(im))}
        if self.dis[i]:
            # DIS/IS-Net/Inference.py: bilinear to 1024, cast to uint8, /255, normalise(0.5, 1.0)
            t = torch.from_numpy(im).float().permute(2, 0, 1)[None]
            t = F.interpolate(t, (1024, 1024), mode='bilinear').type(torch.uint8)[0]
            out['dis_in'] = t
            out['shape'] = torch.tensor(im.shape[:2])
        else:
            out['dis_in'] = torch.zeros(3, 1024, 1024, dtype=torch.uint8)
            out['shape'] = torch.tensor([0, 0])
        return out


def collate(b):
    return {k: torch.stack([x[k] for x in b]) if k != 'i' else torch.tensor([x['i'] for x in b]) for k in b[0]}


def rank_info():
    if 'RANK' in os.environ:
        dist.init_process_group('gloo')
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def barrier(ws):
    if ws > 1:
        dist.barrier()


@torch.no_grad()
def run_split(paths, dis_flags, img_mm, fg_mm, net, rank, ws, dev):
    idx = list(range(rank, len(paths), ws))
    ds = torch.utils.data.Subset(Items(paths, dis_flags), idx)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=16, collate_fn=collate)
    for n, b in enumerate(dl):
        ii = b['i'].numpy()
        img_mm[ii] = b['small'].numpy()
        if fg_mm is None:
            continue
        sel = [k for k in range(len(ii)) if dis_flags[ii[k]]]
        if not sel:
            continue
        x = normalize(b['dis_in'][sel].to(dev).float() / 255.0, [0.5] * 3, [1.0] * 3)
        pred = net(x)[0][0]                                      # (B,1,1024,1024) sigmoid map
        for j, k in enumerate(sel):
            h, w = b['shape'][k].tolist()
            r = F.interpolate(pred[j:j + 1], (h, w), mode='bilinear')[0, 0]
            r = (r - r.min()) / (r.max() - r.min())
            m = ((r > 0.5).to(torch.uint8) * 255).cpu().numpy()  # binarised DISthresh mask
            fg_mm[ii[k]] = resize(m)
        if rank == 0 and n % 50 == 0:
            print(f'  {n}/{len(dl)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = ap.parse_args()
    rank, ws = rank_info()
    dev = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(dev)
    os.chdir(ROOT)
    os.makedirs(CACHE, exist_ok=True)

    from models import ISNetDIS
    net = ISNetDIS()
    net.load_state_dict(torch.load(os.path.join(AUX, 'DIS/saved_models/IS-Net/isnet-general-use.pth'), map_location='cpu'))
    net.to(dev).eval()

    # -- shared: DTD anomaly sources and texture foregrounds
    if rank == 0 and not os.path.exists(os.path.join(CACHE, 'dtd.npy')):
        dtd = sorted(glob.glob(os.path.join(AUX, 'dtd/images/*/*.jpg')))   # same glob as the original
        np.save(os.path.join(CACHE, 'dtd.npy'), np.stack([resize(read_rgb(p)) for p in dtd]))
        tex = sorted(glob.glob(os.path.join(AUX, 'ref/carpet/thresh/*.png')))   # all 32 released masks
        np.save(os.path.join(CACHE, 'texture_fg.npy'), np.stack([resize(cv2.imread(p, 0)) for p in tex]))
        print(f'dtd {len(dtd)} images, texture fg {len(tex)} masks', flush=True)

    for key in args.datasets:
        cfg = DATASETS[key]
        out = os.path.join(CACHE, key)
        os.makedirs(out, exist_ok=True)
        tr = build_dataset(train=True, img_size=S, transform_type='default', **cfg)
        te = build_dataset(train=False, img_size=S, transform_type='default', **cfg)
        names = [d.category for d in tr.datasets]
        assert names == [d.category for d in te.datasets]
        tr_paths = [str(p) for d in tr.datasets for p in d.img_files]
        tr_cls = np.concatenate([np.full(len(d), c) for c, d in enumerate(tr.datasets)]).astype(np.int16)
        te_paths = [str(p) for d in te.datasets for p in d.img_files]
        te_cls = np.concatenate([np.full(len(d), c) for c, d in enumerate(te.datasets)]).astype(np.int16)
        dis_flags = [names[c] not in TEXTURES for c in tr_cls]
        if rank == 0:
            print(f'[{key}] {len(names)} classes, train {len(tr_paths)}, test {len(te_paths)}', flush=True)
            open_memmap(f'{out}/train_img.npy', 'w+', np.uint8, (len(tr_paths), S, S, 3))
            open_memmap(f'{out}/train_fg.npy', 'w+', np.uint8, (len(tr_paths), S, S))
            open_memmap(f'{out}/test_img.npy', 'w+', np.uint8, (len(te_paths), S, S, 3))
            open_memmap(f'{out}/test_mask.npy', 'w+', np.uint8, (len(te_paths), S, S))
            np.save(f'{out}/train_cls.npy', tr_cls)
            np.save(f'{out}/test_cls.npy', te_cls)
            np.save(f'{out}/test_label.npy', np.array([l for d in te.datasets for l in d.labels], dtype=np.uint8))
            json.dump({'classes': names, 'texture': [n in TEXTURES for n in names],
                       'train_paths': tr_paths, 'test_paths': te_paths}, open(f'{out}/meta.json', 'w'))
        barrier(ws)
        img_mm = np.load(f'{out}/train_img.npy', mmap_mode='r+')
        fg_mm = np.load(f'{out}/train_fg.npy', mmap_mode='r+')
        run_split(tr_paths, dis_flags, img_mm, fg_mm, net, rank, ws, dev)
        img_mm.flush(); fg_mm.flush()
        # test images (DiffusionAD cv2 pipeline) and FlowAD GT masks
        timg = np.load(f'{out}/test_img.npy', mmap_mode='r+')
        run_split(te_paths, [False] * len(te_paths), timg, None, net, rank, ws, dev)
        timg.flush()
        tmask = np.load(f'{out}/test_mask.npy', mmap_mode='r+')
        sub = torch.utils.data.Subset(te, list(range(rank, len(te), ws)))
        dl = torch.utils.data.DataLoader(sub, batch_size=64, num_workers=16)
        for b in dl:
            tmask[b['index'].numpy()] = (b['mask'].reshape(-1, S, S) > 0).numpy().astype(np.uint8)
        tmask.flush()
        barrier(ws)
        if rank == 0:
            print(f'[{key}] done', flush=True)


if __name__ == '__main__':
    main()
