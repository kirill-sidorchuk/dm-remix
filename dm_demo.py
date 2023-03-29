"""
    Script to demonstrate the use of stable diffusion model.

    Command line arguments:
    dm_demo.py <prompt> <num_images_in_batch> <image_weights>
"""
import argparse
import os

import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler

from remix_pipe import RemixPipeline

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


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        sd_dtype = torch.bfloat16
        variation_dtype = "fp32"
    else:
        sd_dtype = torch.float16
        variation_dtype = "fp16"

    prompt = [args.prompt]

    # loading image with PIL
    images = []
    for image_file in args.images:
        image = Image.open(image_file).convert("RGB")
        # resize image to {WIDTH, HEIGHT}
        image = image.resize((WIDTH, HEIGHT))
        images.append(image)

    if args.generate_video:
        # generating interpolation video
        if len(images) != 2:
            print("Generating video requires exactly 2 images")
            return

        # creating and cleaning video directory
        os.makedirs("video_dir", exist_ok=True)
        files = os.listdir("video_dir")
        for f in files:
            os.remove(os.path.join("video_dir", f))

        all_image_weights = []
        scale_offset = (1 - args.interpolation_scale) / 2
        for i in range(args.num_frames):
            w1 = args.interpolation_scale * i / (args.num_frames - 1) + scale_offset
            w0 = 1 - w1
            all_image_weights.append([w0, w1])
    else:
        # generating single image
        image_weights = args.image_weights
        if len(image_weights) == 0:
            image_weights = None
        elif len(image_weights) != len(images):
            print("Number of image weights must be equal to number of images")
            return
        all_image_weights = [image_weights]

    # model_id = "stabilityai/stable-diffusion-2-1"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=sd_dtype)
    pipe = RemixPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip",
                                         torch_dtype=sd_dtype,
                                         variation=variation_dtype)

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    with torch.inference_mode():
        for i, image_weights in enumerate(all_image_weights):
            print(f"Generating image {i + 1}/{len(all_image_weights)}")

            # create torch random generator
            generator = torch.Generator().manual_seed(args.seed)

            gen_images = pipe(
                prompt=prompt,
                num_images_per_prompt=args.num_images_in_batch,
                images=images,
                image_weights=image_weights,
                negative_prompt=[args.negative_prompt],
                num_inference_steps=args.num_inference_steps,
                height=HEIGHT, width=WIDTH,
                generator=generator,
                noise_level=args.noise_level,
                timestep=args.timestep,
                start_from_content_latents=args.start_from_content_latents,
            ).images

            grid = image_grid(gen_images, rows=2, cols=2)
            if args.generate_video:
                grid.save("video_dir/remix_%03d.png" % i)
            else:
                grid.save("remix.png")

    if args.generate_video:
        os.system("ffmpeg -y -r %d -i video_dir/remix_%%03d.png -vcodec libx264 -pix_fmt yuv420p remix.mp4" % args.fps)


if __name__ == "__main__":
    # parsing command line args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("images", type=str, help="image file names", nargs="+")
    argparser.add_argument("-p", "--prompt", type=str, default="")
    argparser.add_argument("-w", "--image_weights", type=float, nargs="+", default=[1, 1])
    argparser.add_argument("-n", "--num_images_in_batch", type=int, default=4)
    argparser.add_argument("-g", "--negative_prompt", type=str,
                           default='ugly, boring, cropped, out of frame, jpeg artifacts, mutated')
    argparser.add_argument("-v", "--generate_video", action="store_true", default=False)
    argparser.add_argument("--num_frames", type=int, default=10)
    argparser.add_argument("--fps", type=int, default=10)
    argparser.add_argument("-s", "--noise_level", type=int, default=0)
    argparser.add_argument("--interpolation_scale", type=float, default=1.0)
    argparser.add_argument("--num_inference_steps", type=int, default=50)
    argparser.add_argument("--seed", type=int, default=41)
    argparser.add_argument("-l", "--start_from_content_latents", action="store_true", default=True)
    argparser.add_argument("-t", "--timestep", type=int, default=0, help="timestep for start when -l is used")
    _args = argparser.parse_args()
    main(_args)
