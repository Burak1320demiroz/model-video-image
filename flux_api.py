import gc
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("flux")


class FluxImageGenerator:
    """
    Lazy-loaded FLUX image generator.
    Loads model on demand to avoid keeping VRAM occupied.
    """

    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-schnell") -> None:
        self.model_id = model_id
        self._pipe: Optional["DiffusionPipeline"] = None

    def _load(self) -> "DiffusionPipeline":
        if self._pipe is None:
            logger.info("loading flux model | model_id=%s | device=%s", self.model_id, DEVICE)
            started = time.perf_counter()
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=DTYPE,
            )
            pipe = pipe.to(DEVICE)
            self._pipe = pipe
            logger.info("flux model loaded | took=%.2fs", time.perf_counter() - started)
        else:
            logger.info("reusing cached flux model")
        return self._pipe

    def unload(self) -> None:
        logger.info("unloading flux model")
        self._pipe = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    def generate(
        self,
        prompt: str,
        output_path: str = "output/image.png",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        height: int = 1024,
        width: int = 1024,
        seed: Optional[int] = None,
    ) -> str:
        logger.info(
            "flux generate start | steps=%d guidance=%.2f size=%dx%d seed=%s prompt_len=%d",
            num_inference_steps,
            guidance_scale,
            width,
            height,
            seed,
            len(prompt),
        )
        started = time.perf_counter()
        pipe = self._load()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output)
        logger.info("flux generate done | output=%s | took=%.2fs", output, time.perf_counter() - started)
        return str(output)
