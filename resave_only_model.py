import torch
import os
path = "/home/acg17237wy/projects/FlowAD/logs/realiad_mc_ditG_enet_0901_audiojack/checkpoints/ckpt_epoch_300.pth"
parent_dir = os.path.dirname(path)
checkpoint = torch.load(path, map_location='cpu')
model_ckpt = checkpoint['model'] if 'model' in checkpoint else checkpoint
new_state_dict = {"model": model_ckpt}

save_path = os.path.join(parent_dir, "ckpt_epoch_300.pth")
torch.save(new_state_dict, save_path)