"""Calibrate IS-Net weights + binarisation threshold against the DiffusionAD
authors' released DISthresh masks (first 50 per class from their Drive)."""
import glob, os, sys
import cv2, numpy as np, torch, torch.nn.functional as F
from torchvision.transforms.functional import normalize
AUX = '/home/acg17237wy/projects/FlowAD/data/diffusionad_aux'
sys.path.insert(0, f'{AUX}/DIS/IS-Net')
from models import ISNetDIS
DATA = '/home/acg17237wy/projects/FlowAD/data'

def src_path(cls, name):
    if cls in ('bottle', 'pill'):
        return f'{DATA}/mvtec_ad/{cls}/train/good/{name}'
    return f'{DATA}/visa/{cls}/Data/Images/Normal/{name}'

@torch.no_grad()
def dis(net, im):  # DIS/IS-Net/Inference.py preprocessing
    t = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)[None]
    t = F.interpolate(t, (1024, 1024), mode='bilinear').type(torch.uint8) / 255.0
    t = normalize(t, [0.5] * 3, [1.0] * 3).cuda()
    r = F.interpolate(net(t)[0][0], im.shape[:2], mode='bilinear')[0, 0]
    return ((r - r.min()) / (r.max() - r.min())).cpu().numpy()

ths = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
for w in ['isnet-general-use', 'isnet']:
    net = ISNetDIS(); net.load_state_dict(torch.load(f'{AUX}/DIS/saved_models/IS-Net/{w}.pth', map_location='cpu')); net.cuda().eval()
    for cls in ['bottle', 'pill', 'candle', 'pcb1']:
        ious = {t: [] for t in ths}
        for rp in sorted(glob.glob(f'{AUX}/ref/{cls}/DISthresh/good/*'))[:20]:
            ref = cv2.imread(rp, 0) > 127
            im = cv2.cvtColor(cv2.imread(src_path(cls, os.path.basename(rp))), cv2.COLOR_BGR2RGB)
            p = dis(net, im)
            for t in ths:
                b = p > t
                ious[t].append((b & ref).sum() / max((b | ref).sum(), 1))
        print(w, cls, ' '.join(f'{t}:{np.mean(v):.3f}' for t, v in ious.items()), flush=True)
