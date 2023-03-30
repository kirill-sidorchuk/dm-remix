# Stable Diffusion Remix Demo

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python dm_demo.py content_image.png style_image.png
```
this will produce a remixed image in the current directory. 'remix.png'

## How it works
Uses StableDiffusion unCLIP together with img2img.
unCLIP is used to create and them mix image embeddings of content and style images.
img2img is used to initialize latents from content image and then run reverse diffusion with unCLIP embeddings as guidance.
Optionally you can condition on a text prompt: '-p' parameter.
Also you can specify mixing weights for content and style images: '-w' parameter. The default weights are [1, 1].


There is a video mode in which 10 frames are generated with the range of mixing weights: '-v' parameter.
Video frames will be generated in 'video_dir' directory and remix.mp4 video will be created in the current directory.
Note you need to have ffmpeg installed to create a video.
```
apt get install ffmpeg
```
