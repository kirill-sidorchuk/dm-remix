from typing import List, Union, Optional, Callable, Dict, Any

import PIL
import numpy as np
import torch
from PIL import Image
from diffusers import StableUnCLIPImg2ImgPipeline, UNet2DConditionModel, AutoencoderKL, ImagePipelineOutput, \
    StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableUnCLIPImageNormalizer
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, randn_tensor, PIL_INTERPOLATION
from transformers import CLIPFeatureExtractor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """
    Symmetric linear interpolation between two vectors on the unit sphere. Batched version.
    See https://en.wikipedia.org/wiki/Slerp,
    https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3

    :param val: interpolation value in [0, 1]
    :param low: the first point, not need to be normalized
    :param high: the second point, not need to be normalized

    :return: the interpolated point
    """

    assert len(low.shape) == len(high.shape) == 2

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def center_resize_crop(image, size=224):
    w, h = image.size
    if h < w:
        h, w = size, size * w // h
    else:
        h, w = size * h // w, size

    image = image.resize((w, h))

    box = ((w - size) // 2, (h - size) // 2, (w + size) // 2, (h + size) // 2)
    return image.crop(box)


class RemixPipeline(StableUnCLIPImg2ImgPipeline):
    """
    Image remixing pipeline using the StableUnCLIPImg2ImgPipeline as a base. This pipeline is used to remix images
    """

    def __init__(
            self,
            feature_extractor: CLIPFeatureExtractor,
            image_encoder: CLIPVisionModelWithProjection,
            image_normalizer: StableUnCLIPImageNormalizer,
            image_noising_scheduler: KarrasDiffusionSchedulers,
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            vae: AutoencoderKL,
    ):
        super().__init__(feature_extractor, image_encoder, image_normalizer, image_noising_scheduler, tokenizer,
                         text_encoder, unet, scheduler, vae)

    def prepare_latents_from_image(
            self,
            image,
            timestep,
            batch_size,
            num_images_per_prompt,
            dtype,
            device,
            generator=None,
            noise=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i: i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0",
                      deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        if noise is None:
            noise = randn_tensor(shape, generator=generator,
                                 device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def _encode_image(
        self,
        images: List[Union[torch.Tensor, PIL.Image.Image]],
        image_weights: List[float],
        device,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance,
        noise_level,
        generator,
        image_embeds,
    ):
        dtype = next(self.image_encoder.parameters()).dtype

        if isinstance(images[0], PIL.Image.Image):
            # the image embedding should repeated so it matches the total batch size of the prompt
            repeat_by = batch_size
        else:
            # assume the image input is already properly batched and just needs to be repeated so
            # it matches the num_images_per_prompt.
            #
            # NOTE(will) this is probably missing a few number of side cases. I.e. batched/non-batched
            # `image_embeds`. If those happen to be common use cases, let's think harder about
            # what the expected dimensions of inputs should be and how we handle the encoding.
            repeat_by = num_images_per_prompt

        if not image_embeds:
            if not isinstance(images[0], torch.Tensor):
                images = self.feature_extractor(images=images, return_tensors="pt").pixel_values

            images = images.to(device=device, dtype=dtype)
            image_embeds = self.image_encoder(images).image_embeds

        # interpolate image embeddings
        assert len(image_embeds) == len(image_weights)

        if image_embeds.shape[0] == 2:
            # using slerp interpolation
            intp_value = image_weights[1] / (image_weights[0] + image_weights[1])
            image_embeds = slerp(intp_value, image_embeds[1].unsqueeze(0), image_embeds[0].unsqueeze(0))
        elif image_embeds.shape[0] > 1:
            # using linear interpolation
            image_weights = torch.tensor(image_weights,
                                         dtype=image_embeds.dtype,
                                         device=image_embeds.device).view(-1, 1, 1)  # [N, 1, 1]
            image_embeds = torch.sum(image_embeds * image_weights, dim=0) / torch.sum(image_weights, dim=0)

        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
        )

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        image_embeds = image_embeds.unsqueeze(1)
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, repeat_by, 1)
        image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
        image_embeds = image_embeds.squeeze(1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeds = torch.cat([negative_prompt_embeds, image_embeds])

        return image_embeds

    @staticmethod
    def _preprocess_image(image: PIL.Image.Image) -> torch.Tensor:
        """
        Taken from StableDiffusionImg2ImgPipeline
        :param image: PIL image
        :return: torch tensor
        """

        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            w, h = image[0].size
            w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

            image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = 2.0 * image - 1.0
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)
        return image

    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            images: List[Union[torch.FloatTensor, PIL.Image.Image]] = None,
            image_weights: Optional[List[float]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 20,
            guidance_scale: float = 10,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            noise_level: int = 0,
            image_embeds: Optional[torch.FloatTensor] = None,
            timestemp: int = 0,
            start_from_content_latents: bool = False
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch. The image will be encoded to its CLIP embedding which
                the unet will be conditioned on. Note that the image is _not_ encoded by the vae and then used as the
                latents in the denoising process such as in the standard stable diffusion text guided image variation
                process.
            image_weights (`List[float]`, *optional*):
                The weight of each image in the list. If not defined, all images will have the same weight.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            noise_level (`int`, *optional*, defaults to `0`):
                The amount of noise to add to the image embeddings. A higher `noise_level` increases the variance in
                the final un-noised images. See `StableUnCLIPPipeline.noise_image_embeddings` for details.
            image_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated CLIP embeddings to condition the unet on. Note that these are not latents to be used in
                the denoising process. If you want to provide pre-generated latents, pass them to `__call__` as
                `latents`.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~ pipeline_utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            image=images[0],
            height=height,
            width=width,
            callback_steps=callback_steps,
            noise_level=noise_level,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = batch_size * num_images_per_prompt

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Encoder input image
        noise_level = torch.tensor([noise_level], device=device)
        if image_weights is None:
            image_weights = [1.0] * len(images)
        image_embeds = self._encode_image(
            images=images,
            image_weights=image_weights,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
            generator=generator,
            image_embeds=None,
        )

        torch.cuda.empty_cache()

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        if start_from_content_latents:
            # using content image to get starting latents
            # diffusing encoded content image
            latent_timestep = timesteps[timestemp:timestemp +
                                                  1].repeat(num_images_per_prompt)

            content_image = self._preprocess_image(images[0])
            # [1, 3, 512, 512]

            # duplicate content_image for each generation per prompt, using mps friendly method
            content_image = content_image.expand(num_images_per_prompt, -1, -1, -1).contiguous()

            latents = self.prepare_latents_from_image(
                image=content_image,
                timestep=latent_timestep,
                batch_size=1,
                dtype=prompt_embeds.dtype,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                generator=generator,
                noise=latents
            )

        else:
            # using random latents
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
                latents=latents,
            )
            # latents: [batch_size, num_channels_latents, height, width]

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 9. Post-processing
        image = self.decode_latents(latents)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
