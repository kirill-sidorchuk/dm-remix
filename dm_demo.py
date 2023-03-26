"""
    Script to demonstrate the use of stable diffusion model.
"""
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline


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


model_id = "stabilityai/stable-diffusion-2-1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
sd_dtype = torch.bfloat16 if device == "cpu" else torch.float16
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=sd_dtype)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = ["a photo of a cat that looks exactly like a loaf bread"] * 4

# loading image with PIL
image = Image.open("the_cat.png")

# resize image to 512x512
image = image.resize((512, 512))

images = pipe(prompt=prompt, image=[image] * len(prompt), strength=0.6).images
grid = image_grid(images, rows=2, cols=2)

grid.save("remix.png")
