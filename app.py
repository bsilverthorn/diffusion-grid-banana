from io import BytesIO
from base64 import (
    b64encode,
    b64decode,
)
from typing import (
    List,
    Optional,
)

import torch
import pydantic

from torch import Tensor
from pydantic import validator
from PIL.Image import Image
from runner import (
    init_model,
    ModelRunner,
)


class Settings(pydantic.BaseSettings):
    hf_auth_token: str
    diffusion_torch_device: str = "cuda"
    diffusion_guidance_scale: float = 7.5
    diffusion_inference_steps: int = 50
    diffusion_scheduler_eta: float = 1.0

settings = Settings.parse_obj({})

def init():
    model = init_model(settings.hf_auth_token, settings.diffusion_torch_device)
    model_runner = ModelRunner(
        model,
        guidance_scale=settings.diffusion_guidance_scale,
        inference_steps=settings.diffusion_inference_steps,
        scheduler_eta=settings.diffusion_scheduler_eta,
        torch_device=settings.diffusion_torch_device,
    )

    return (model, model_runner)

def tensor_to_base64(tensor: Tensor) -> str:
    buffered = BytesIO()

    torch.save(tensor, buffered)

    return b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_tensor(encoded: str) -> Tensor:
    buffered = BytesIO(b64decode(encoded))

    return torch.load(buffered, map_location=settings.diffusion_torch_device)

def image_to_base64(image: Image, format="JPEG") -> str:
    buffered = BytesIO()

    image.save(buffered, format=format)

    return b64encode(buffered.getvalue()).decode("utf-8")

class ModelIO(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Tensor: tensor_to_base64,
            Image: image_to_base64,
        }

class ModelRunInputs(ModelIO):
    prompt: str
    seed: int
    latents: Optional[Tensor]
    timestep: Optional[int]
    trajectory_at: List[int]

    @validator("latents", pre=True)
    def latents_from_base64(cls, v):
        return None if v is None else base64_to_tensor(v)

class ModelInputs(ModelIO):
    cache_key: str
    run_inputs: ModelRunInputs

class ModelLatentsInfo(ModelIO):
    tensor: Tensor
    image: Image
    timestep: int

class ModelRunOutputs(ModelIO):
    image: Image
    trajectory: List[ModelLatentsInfo]

class ModelOutputs(ModelIO):
    prompt: str
    cache_key: str
    run_outputs: ModelRunOutputs

def inference(model_inputs: dict, model_runner: ModelRunner) -> str:
    typed_inputs = ModelInputs.parse_obj(model_inputs)

    return inference_typed(typed_inputs, model_runner).json()

def inference_typed(model_inputs: ModelInputs, model_runner: ModelRunner) -> ModelOutputs:
    """Run the diffusion pipeline, optionally starting from a latents tensor."""

    (generated, trajectory) = model_runner.run(**model_inputs.run_inputs.dict())
    (image,) = generated.images

    return ModelOutputs(
        prompt=model_inputs.run_inputs.prompt,
        cache_key=model_inputs.cache_key,
        run_outputs=ModelRunOutputs(
            image=image,
            trajectory=[
                ModelLatentsInfo(
                    tensor=latents,
                    image=model_runner.latents_to_image(latents),
                    timestep=t,
                )
                for (t, latents) in trajectory.items()
            ],
        ),
    )
