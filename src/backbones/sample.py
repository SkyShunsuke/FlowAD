import torch
from transformers import SiglipVisionModel, AutoImageProcessor

model_id = "google/siglip2-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_id)
model = SiglipVisionModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",   # 対応環境ならflash_attention_2でも可
).eval().cuda()

image = torch.rand(1, 3, 224, 224).cuda()  # (B, C, H, W)
inputs = processor(images=image, return_tensors="pt").to("cuda", torch.bfloat16)
print(inputs["pixel_values"].shape)
print(processor)

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

# base: hidden_states は 13要素 (embedding + 12ブロック), 各 (B, 196, 768)
feat = out.hidden_states[9]

B, L, D = feat.shape
h = w = int(L ** 0.5)   # 224/16 = 14 → 14×14
feat_2d = feat.permute(0, 2, 1).reshape(B, D, h, w)
print(f"feat_2d: {feat_2d.shape}")  # (B, D, h, w) = (1, 768, 14, 14)