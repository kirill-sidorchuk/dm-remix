"""
    Script to demonstrate the use of stable diffusion model.
"""
import sys

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, \
    StableUnCLIPImg2ImgPipeline

from remix_pipe import RemixPipeline

WIDTH = 512
HEIGHT = 512
num_images_in_batch = 2


def image_grid(imgs, rows=2, cols=2):
    """Create a grid of images.

    :param imgs: list of PIL images
    :param rows: number of rows
    :param cols: number of columns
    """
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    sd_dtype = torch.bfloat16
    variation_dtype = "fp32"
else:
    sd_dtype = torch.float16
    variation_dtype = "fp16"

# model_id = "stabilityai/stable-diffusion-2-1"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=sd_dtype)

pipe = RemixPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip",
                                     torch_dtype=sd_dtype,
                                     variation=variation_dtype)

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

if len(sys.argv) > 1:
    prompt = sys.argv[1]
else:
    prompt = "a photo of a creature"
print('prompt: ', prompt)

if len(sys.argv) > 2:
    num_images_in_batch = int(sys.argv[2])
print('num_images_in_batch: ', num_images_in_batch)

prompt = [prompt] * num_images_in_batch

# loading image with PIL
image1 = Image.open("the_cat.png")
image2 = Image.open("the_bread.png")

# resize image to {WIDTH, HEIGHT}
image1 = image1.resize((WIDTH, HEIGHT))
image2 = image2.resize((WIDTH, HEIGHT))

# create torch random generator
generator = torch.Generator().manual_seed(41)

# image=[image] * len(prompt), strength=0.9
with torch.inference_mode():
    images = pipe(prompt=prompt,
                  images=[image1, image2],
                  negative_prompt=['ugly, boring, cropped, out of frame, jpeg artifacts'] * len(prompt),
                  num_inference_steps=50,
                  height=HEIGHT, width=WIDTH,
                  generator=generator).images
grid = image_grid(images, rows=2, cols=2)

grid.save("remix.png")
