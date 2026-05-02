import gc
import logging
import time
import base64
import requests
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, Dict, List

import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("flux")

def load_image(ref_str: str) -> Image.Image:
    """Load an image from a base64 data URL, HTTP URL, or local path."""
    if ref_str.startswith("data:image"):
        _, encoded = ref_str.split(",", 1)
        data = base64.b64decode(encoded)
        return Image.open(BytesIO(data)).convert("RGB")
    elif ref_str.startswith("http"):
        response = requests.get(ref_str, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(ref_str).convert("RGB")


class FluxImageGenerator:
    """
    Lazy-loaded FLUX image generator.
    Loads model on demand to avoid keeping VRAM occupied.
    """

    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-dev") -> None:
        self.model_id = model_id
        self._pipe: Optional["DiffusionPipeline"] = None

    def _configure_pipe_memory(self, pipe: "DiffusionPipeline") -> None:
        if DEVICE != "cuda":
            pipe.to(DEVICE)
            return

        # 24GB cards can OOM while moving the full FLUX pipeline to CUDA.
        # CPU offload keeps only active components on GPU.
        try:
            pipe.enable_model_cpu_offload()
            logger.info("flux cpu offload enabled")
        except Exception:
            logger.warning("flux cpu offload unavailable, falling back to full cuda move")
            pipe.to(DEVICE)

        # Extra memory savers for high resolutions.
        try:
            pipe.enable_attention_slicing("max")
            logger.info("flux attention slicing enabled")
        except Exception:
            logger.warning("flux attention slicing not available")
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            logger.info("flux vae tiling/slicing enabled")
        except Exception:
            logger.warning("flux vae tiling/slicing not available")

    def _load(self, use_controlnet: bool = False) -> "DiffusionPipeline":
        # If pipeline is loaded but we need a different class (ControlNet vs base), 
        # a full implementation would swap it. For now, we will load the base pipe 
        # and leave placeholders for FluxControlNetPipeline integration.
        if self._pipe is None:
            logger.info("loading flux model | model_id=%s | device=%s", self.model_id, DEVICE)
            started = time.perf_counter()
            
            if use_controlnet:
                logger.info("ControlNet requested - loading FluxControlNetPipeline (Placeholder)")
                from diffusers import FluxControlNetPipeline, FluxControlNetModel
                # Replace 'your-controlnet-repo' with actual FLUX Canny/Depth ControlNet
                # controlnet = FluxControlNetModel.from_pretrained("Shakker-Labs/FLUX.1-dev-ControlNet-Depth", torch_dtype=DTYPE)
                # pipe = FluxControlNetPipeline.from_pretrained(self.model_id, controlnet=controlnet, torch_dtype=DTYPE)
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=DTYPE)
            else:
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=DTYPE,
                )
            
            self._configure_pipe_memory(pipe)
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
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        height: int = 480,
        width: int = 832,
        seed: Optional[int] = None,
        character_references: Optional[List[Dict[str, Any]]] = None,
        environment_reference: Optional[Dict[str, Any]] = None,
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
        
        has_env = environment_reference is not None
        pipe = self._load(use_controlnet=has_env)
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # Build kwargs for the pipeline
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "generator": generator,
        }

        # Handle IP-Adapter (Character Consistency)
        if character_references:
            logger.info("Processing %d character reference(s) for IP-Adapter", len(character_references))
            # Placeholder for IP-Adapter injection
            # pipe.load_ip_adapter(...)
            # pipe.set_ip_adapter_scale(...)
            # kwargs["ip_adapter_image"] = [load_image(ref["reference_image"]) for ref in character_references]
        
        # Handle ControlNet (Environment Consistency)
        if environment_reference:
            logger.info("Processing environment reference: %s", environment_reference.get("name"))
            env_img = load_image(environment_reference["reference_image"])
            # Assuming Canny or Depth preprocessing would go here
            # kwargs["control_image"] = env_img
            # kwargs["controlnet_conditioning_scale"] = environment_reference.get("structure_strength", 0.7)

        result = pipe(**kwargs)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output)
        logger.info("flux generate done | output=%s | took=%.2fs", output, time.perf_counter() - started)
        return str(output)
