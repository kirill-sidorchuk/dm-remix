"""
    Script to demonstrate the use of stable diffusion model.
"""
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on the moon during a comet shower"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")
