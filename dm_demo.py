"""
    Script to demonstrate the use of stable diffusion model.

    Command line arguments:
    dm_demo.py <prompt> <num_images_in_batch> <image_weights>
"""
import argparse
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


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        sd_dtype = torch.bfloat16
        variation_dtype = "fp32"
    else:
        sd_dtype = torch.float16
        variation_dtype = "fp16"

    prompt = [args.prompt] * args.num_images_in_batch

    # loading image with PIL
    images = []
    for image_file in args.images:
        image = Image.open(image_file)
        # resize image to {WIDTH, HEIGHT}
        image = image.resize((WIDTH, HEIGHT))
        images.append(image)

    image_weights = args.image_weights
    if len(image_weights) == 0:
        image_weights = None
    elif len(image_weights) != len(images):
        print("Number of image weights must be equal to number of images")
        return

    # model_id = "stabilityai/stable-diffusion-2-1"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=sd_dtype)
    pipe = RemixPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip",
                                         torch_dtype=sd_dtype,
                                         variation=variation_dtype)

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # create torch random generator
    generator = torch.Generator().manual_seed(41)

    # image=[image] * len(prompt), strength=0.9
    with torch.inference_mode():
        images = pipe(prompt=prompt,
                      images=images,
                      image_weights=image_weights,
                      negative_prompt=['ugly, boring, cropped, out of frame, jpeg artifacts'] * len(prompt),
                      num_inference_steps=50,
                      height=HEIGHT, width=WIDTH,
                      generator=generator).images
    grid = image_grid(images, rows=2, cols=2)

    grid.save("remix.png")


if __name__ == "__main__":
    # parsing command line args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-p", "--prompt", type=str, default="")
    argparser.add_argument("-i", "--images", type=str, help="image file names", nargs="+", required=True)
    argparser.add_argument("-w", "--image_weights", type=float, nargs="+", default=[])
    argparser.add_argument("-n", "--num_images_in_batch", type=int, default=4)
    _args = argparser.parse_args()
    main(_args)
