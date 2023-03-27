"""
    Script to demonstrate the use of stable diffusion model.
"""
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, \
    StableUnCLIPImg2ImgPipeline

from remix_pipe import RemixPipeline

WIDTH = 512
HEIGHT = 768
num_images_in_batch = 1


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

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
sd_dtype = torch.bfloat16 if device == "cpu" else torch.float16

# model_id = "stabilityai/stable-diffusion-2-1"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=sd_dtype)

pipe = RemixPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip",
                                     torch_dtype=sd_dtype)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = ["a photo of a creature"] * num_images_in_batch

# loading image with PIL
image1 = Image.open("the_cat.png")
image2 = Image.open("the_bread.png")

# resize image to {WIDTH, HEIGHT}
image1 = image1.resize((WIDTH, HEIGHT))
image2 = image2.resize((WIDTH, HEIGHT))

# image=[image] * len(prompt), strength=0.9
with torch.inference_mode():
    images = pipe(prompt=prompt,
                  images=[image1, image2],
                  negative_prompt=['ugly, boring'] * len(prompt),
                  num_inference_steps=50,
                  height=HEIGHT, width=WIDTH).images
grid = image_grid(images, rows=1, cols=2)

grid.save("remix.png")
