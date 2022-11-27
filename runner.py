from contextlib import nullcontext
from typing import (
    Any,
    Tuple,
    Dict,
    Set,
    Optional,
)

from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from torch import (
    Tensor,
    Generator,
)
from torch.amp.autocast_mode import autocast
from PIL.Image import Image


class FastForwardDDIMScheduler(DDIMScheduler):
    forward_to: Optional[int] = None

    def set_timesteps(self, *args, **kwargs):
        super().set_timesteps(*args, **kwargs)  # type: ignore 

        if self.forward_to is not None:
            self.timesteps = self.timesteps[self.timesteps < self.forward_to]

class ModelRunner:
    def __init__(
            self,
            model,
            guidance_scale: float,
            inference_steps: int,
            scheduler_eta: float,
            torch_device: str):
        self._model = model
        self._guidance_scale = guidance_scale
        self._inference_steps = inference_steps
        self._scheduler_eta = scheduler_eta
        self._torch_device = torch_device

    def run(
            self,
            prompt: str,
            seed: int,
            trajectory_at: Set[int] = set(),
            latents: Optional[Tensor] = None,
            timestep: Optional[int] = None) -> Tuple[Any, Dict[int, Tensor]]:
        # prep the prng
        if seed is not None:
            generator = Generator(self._torch_device).manual_seed(seed)
        else:
            generator = None

        # prepare to capture diffusion trajectory
        trajectory = {}

        def callback(i, t, latents):
            if t in trajectory_at:
                trajectory[int(t)] = latents

        # run diffusion process
        self._model.scheduler.forward_to = timestep

        if self._torch_device == "cuda":
            autocasting = autocast(self._torch_device)
        else:
            autocasting = nullcontext()

        with autocasting:
            generated = self._model(
                prompt.lower(),
                guidance_scale=self._guidance_scale,
                callback=callback,
                latents=latents,
                eta=self._scheduler_eta,
                height=512,
                width=512,
                generator=generator,
                num_inference_steps=self._inference_steps,
            )

        return (generated, trajectory)

    def latents_to_image(self, latents: Tensor) -> Image:
        latents = 1 / 0.18215 * latents.detach()

        with autocast("cuda"):
            image = self._model.vae.decode(latents).sample.detach()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).float().cpu().numpy()
        (pil_image,) = self._model.numpy_to_pil(image)

        return pil_image

def init_model(hf_auth_token: str, torch_device: str) -> StableDiffusionPipeline:
    scheduler = FastForwardDDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        scheduler=scheduler,
        use_auth_token=hf_auth_token,
    ).to(torch_device)  # type: ignore

    return model
