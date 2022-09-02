# %%
import torch
from diffusers import DiffusionPipeline

model_id = "CompVis/ldm-text2im-large-256"

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id)

# device = "cuda"
# ldm = ldm.to(device)
# %%
# run pipeline in inference (sample random noise and denoise)
prompt = "総書記, ボタン, missile"
images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)["sample"]

# save images

for idx, image in enumerate(images):
    image.save(f"{prompt.split(' ')[0]}-{idx}.png")
# %%


# %%
