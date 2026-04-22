import gc
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusers import LTX2ImageToVideoPipeline, LTXImageToVideoPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("ltx")


class LtxVideoGenerator:
    """
    LTX image-to-video generator.
    Tries low-VRAM LTX-2.3 nvfp4 first, then falls back to LTX-Video.
    """

    def __init__(
        self,
        model_id: str = "Lightricks/LTX-Video",
        fallback_model_id: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.fallback_model_id = fallback_model_id
        self._pipe: Optional["LTX2ImageToVideoPipeline | LTXImageToVideoPipeline"] = None
        self._active_model_id: Optional[str] = None
        self._using_ltx2 = False

    @staticmethod
    def _snap_to_32(value: int) -> int:
        return max(32, (value // 32) * 32)

    @staticmethod
    def _snap_frames(value: int) -> int:
        # LTX family works best with 8n+1 frame counts.
        if value < 9:
            return 9
        remainder = (value - 1) % 8
        return value if remainder == 0 else value + (8 - remainder)

    @staticmethod
    def _compress_prompt(prompt: str, max_words: int = 100) -> str:
        words = prompt.split()
        if len(words) <= max_words:
            return prompt
        return " ".join(words[:max_words])

    def _configure_pipe_memory(self, pipe: "LTX2ImageToVideoPipeline | LTXImageToVideoPipeline") -> None:
        if DEVICE != "cuda":
            pipe.to(DEVICE)
            return

        try:
            pipe.enable_model_cpu_offload()
            logger.info("ltx cpu offload enabled for lower vram")
        except Exception:
            logger.warning("ltx cpu offload unavailable, moving model to cuda directly")
            pipe.to(DEVICE)

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)

        # Accept torch-like BCHW/CHW/HWC frame layouts and convert to HWC.
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)

        if arr.ndim != 3:
            raise ValueError(f"unsupported frame shape: {arr.shape}")

        # Convert float outputs [0,1] to uint8.
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.nanmax(arr)) if arr.size else 0.0
            if max_val <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Ensure channel count is imageio compatible.
        if arr.shape[-1] not in (1, 2, 3, 4):
            raise ValueError(f"invalid channel count: {arr.shape[-1]} for frame shape {arr.shape}")
        return arr

    def _load_ltx2_pipe(self, model_id: str) -> "LTX2ImageToVideoPipeline":
        # LTX2 pipeline location differs across diffusers versions.
        try:
            from diffusers import LTX2ImageToVideoPipeline
        except ImportError:
            from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline

        logger.info("loading ltx2 image-to-video | model_id=%s | device=%s", model_id, DEVICE)
        pipe = LTX2ImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self._configure_pipe_memory(pipe)
        self._using_ltx2 = True
        return pipe

    def _load_ltx_pipe(self, model_id: str) -> "LTXImageToVideoPipeline":
        from diffusers import LTXImageToVideoPipeline

        logger.info("loading ltx image-to-video | model_id=%s | device=%s", model_id, DEVICE)
        pipe = LTXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self._configure_pipe_memory(pipe)
        self._using_ltx2 = False
        return pipe

    def _load(self) -> "LTX2ImageToVideoPipeline | LTXImageToVideoPipeline":
        if self._pipe is not None:
            logger.info("reusing cached ltx model | model_id=%s", self._active_model_id)
            return self._pipe

        started = time.perf_counter()
        load_errors = []
        candidates = [self.model_id]
        if self.fallback_model_id and self.fallback_model_id != self.model_id:
            candidates.append(self.fallback_model_id)

        for idx, candidate in enumerate(candidates):
            try:
                # Prefer LTX2 pipeline for LTX-2/2.3 model IDs.
                if "LTX-2" in candidate or "ltx-2" in candidate:
                    pipe = self._load_ltx2_pipe(candidate)
                else:
                    pipe = self._load_ltx_pipe(candidate)
                self._pipe = pipe
                self._active_model_id = candidate
                logger.info(
                    "ltx model loaded | active_model=%s | ltx2=%s | took=%.2fs",
                    candidate,
                    self._using_ltx2,
                    time.perf_counter() - started,
                )
                return pipe
            except Exception as exc:
                load_errors.append(f"{candidate}: {exc}")
                logger.exception("ltx model load failed | candidate=%s", candidate)
                if idx == 0 and len(candidates) > 1:
                    logger.warning("trying fallback ltx model")

        raise RuntimeError(f"all ltx model load attempts failed: {' | '.join(load_errors)}")

    def unload(self) -> None:
        logger.info("unloading ltx model | active_model=%s", self._active_model_id)
        self._pipe = None
        self._active_model_id = None
        self._using_ltx2 = False
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    def generate(
        self,
        image_path: str,
        prompt: str,
        output_path: str = "output/video.mp4",
        fps: int = 20,
        frames: int = 65,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.2,
    ) -> str:
        """
        Generate video from a source image with LTX.
        """
        logger.info(
            "ltx generate start | image=%s fps=%d frames=%d steps=%d guidance=%.2f prompt_len=%d",
            image_path,
            fps,
            frames,
            num_inference_steps,
            guidance_scale,
            len(prompt),
        )
        started = time.perf_counter()
        pipe = self._load()

        base = Image.open(image_path).convert("RGB")
        raw_w, raw_h = base.size
        width = self._snap_to_32(raw_w)
        height = self._snap_to_32(raw_h)
        num_frames = self._snap_frames(frames)
        if (width, height) != (raw_w, raw_h):
            base = base.resize((width, height), Image.BICUBIC)
        logger.info(
            "ltx normalized params | size=%dx%d->%dx%d frames=%d->%d model=%s",
            raw_w,
            raw_h,
            width,
            height,
            frames,
            num_frames,
            self._active_model_id,
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        generator = None
        if DEVICE == "cuda":
            generator = torch.Generator(device=DEVICE).manual_seed(int(time.time()) % 1_000_000)

        safe_prompt = self._compress_prompt(prompt, max_words=100)
        if safe_prompt != prompt:
            logger.info("ltx prompt truncated | original_words=%d kept_words=%d", len(prompt.split()), len(safe_prompt.split()))

        try:
            result = pipe(
                image=base,
                prompt=safe_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=float(fps),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="np",
            )
        except TypeError:
            # Backward compatibility for older signatures without frame_rate.
            result = pipe(
                image=base,
                prompt=safe_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="np",
            )

        frames_np = None
        if hasattr(result, "frames"):
            frames_np = result.frames[0] if isinstance(result.frames, list) else result.frames
        elif isinstance(result, tuple) and len(result) > 0:
            frames_np = result[0][0] if isinstance(result[0], list) else result[0]

        if frames_np is None:
            raise RuntimeError("ltx pipeline returned no frames")

        # Some pipelines return (batch, frames, H, W, C) and some return (frames, H, W, C).
        if isinstance(frames_np, np.ndarray) and frames_np.ndim == 5:
            frames_np = frames_np[0]
        elif isinstance(frames_np, list) and len(frames_np) == 1 and isinstance(frames_np[0], np.ndarray) and frames_np[0].ndim == 4:
            frames_np = frames_np[0]

        logger.info("ltx writing video | frames=%d fps=%d", len(frames_np), fps)
        writer = imageio.get_writer(str(output), fps=fps, codec="libx264")
        try:
            for i, frame in enumerate(frames_np):
                normalized = self._normalize_frame(frame)
                writer.append_data(normalized)
                if i == 0 or (i + 1) % 12 == 0 or i == len(frames_np) - 1:
                    logger.info(
                        "ltx frame progress | frame=%d/%d shape=%s",
                        i + 1,
                        len(frames_np),
                        normalized.shape,
                    )
        finally:
            writer.close()

        logger.info("ltx generate done | output=%s | took=%.2fs", output, time.perf_counter() - started)
        return str(output)
