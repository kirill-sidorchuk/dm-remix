"""
    Script to demonstrate the use of stable diffusion model.
"""
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

WIDTH = 512
HEIGHT = 512


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
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=sd_dtype)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = ["a photo of a creature that is a mix of a cat and a loaf of bread"] * 2

# loading image with PIL
image = Image.open("the_cat.png")

# resize image to 768x512
image = image.resize((WIDTH, HEIGHT))

# image=[image] * len(prompt), strength=0.9
images = pipe(prompt=prompt,
              negative_prompt=['ugly, boring, bad anatomy'] * len(prompt),
              num_inference_steps=100,
              height=HEIGHT, width=WIDTH).images
grid = image_grid(images, rows=1, cols=2)

grid.save("remix.png")
