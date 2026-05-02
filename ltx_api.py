import gc
import inspect
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import imageio.v2 as imageio
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from PIL import Image

if TYPE_CHECKING:
    from diffusers import LTX2ImageToVideoPipeline, LTXImageToVideoPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("ltx")

# Defaults tuned for high-end GPUs (e.g. RTX 4090). Lower with env or request fields if OOM.
# LTX_MAX_FRAMES / LTX_MIN_* / LTX_CPU_OFFLOAD override behaviour.
DEFAULT_VIDEO_FPS = 24.0
# Daha akıcı bir video için kare sayısını 161 (yaklaşık 6.7 saniye) yapıyoruz.
DEFAULT_VIDEO_FRAMES = 161
DEFAULT_INFERENCE_STEPS = 40
# Yüksek guidance scale (CFG), LTX'te yüzlerin erimesine (morphing) ve deformasyona sebep olur. O yüzden 3.0'a indiriyoruz.
DEFAULT_GUIDANCE_SCALE = 3.0
# DiT modellerinde uzun negatif prompt modelin dikkatini bozar, boş bırakıyoruz.
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_MAX_FRAMES_CAP = 1209


def _filter_pipeline_kwargs(pipe: Any, kw: dict) -> dict:
    """Forward only kwargs supported by this diffusers pipe (LTX vs LTX-2 differ)."""
    try:
        sig = inspect.signature(pipe.__call__)
        params = sig.parameters.keys()
        return {k: v for k, v in kw.items() if k in params}
    except (TypeError, ValueError):
        return kw


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
    def _compress_prompt(prompt: str, max_words: int = 60) -> str:
        words = prompt.split()
        if len(words) <= max_words:
            return prompt
        return " ".join(words[:max_words])

    @staticmethod
    def _upscale_target_size(raw_w: int, raw_h: int, min_w: Optional[int] = None, min_h: Optional[int] = None) -> tuple[int, int]:
        # LTX-Video 832x480 çözünürlüğünde sweet spot'tadır.
        mw = int(os.getenv("LTX_MIN_WIDTH", "832")) if min_w is None else int(min_w)
        mh = int(os.getenv("LTX_MIN_HEIGHT", "480")) if min_h is None else int(min_h)
        mw = max(256, (mw // 32) * 32)
        mh = max(256, (mh // 32) * 32)
        
        # Eğer orijinal resim LTX'in sevdiği boyuttan büyükse, AŞAĞI ÖLÇEKLENDİR. (Downscale)
        # Önceki kod sadece upscale ediyordu, bu da 1024x576 çözünürlüğünde modelin saçmalamasına yol açıyordu.
        scale_w = mw / max(raw_w, 1)
        scale_h = mh / max(raw_h, 1)
        scale = min(scale_w, scale_h, 1.0) # Hem upscale hem downscale yapmasını engellemiyoruz, sadece sınıra oturtuyoruz.
        
        return int(raw_w * scale), int(raw_h * scale)

    def _configure_pipe_memory(self, pipe: "LTX2ImageToVideoPipeline | LTXImageToVideoPipeline") -> None:
        if DEVICE != "cuda":
            pipe.to(DEVICE)
            return

        # RTX 5090'da bile 241 kare (8s video) VAE aşamasında çöktüğü için CPU Offload ZORUNLU
        use_offload = True
        if use_offload:
            try:
                pipe.enable_model_cpu_offload()
                logger.info("ltx cpu offload enabled (LTX_CPU_OFFLOAD=1)")
            except Exception:
                logger.warning("ltx cpu offload unavailable, moving model to cuda directly")
                pipe.to(DEVICE)
        else:
            pipe.to(DEVICE)
            logger.info("ltx full-GPU load (set LTX_CPU_OFFLOAD=1 if you hit CUDA OOM)")

        # ÇOK KRİTİK: VAE Slicing ve Tiling açılmazsa 241 kare videoyu decode ederken 5090 bile çöker
        try:
            if hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
            elif hasattr(pipe.vae, 'enable_slicing'):
                pipe.vae.enable_slicing()
            logger.info("ltx vae slicing enabled")
        except Exception as e:
            logger.warning(f"vae slicing could not be enabled: {e}")
            
        # VAE Tiling video birleşim yerlerinde çamurlaşma yapar, O YÜZDEN KAPATILDI.
        # Sadece Slicing açık kalıyor, 5090'da CPU Offload + Slicing OOM'i önlemeye yeterlidir.

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
        fps: Optional[float] = None,
        frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate video from a source image with LTX.
        """
        eff_fps = DEFAULT_VIDEO_FPS if fps is None else float(fps)
        eff_fps = max(8.0, min(60.0, eff_fps))
        eff_frames = DEFAULT_VIDEO_FRAMES if frames is None else int(frames)
        eff_frames = max(9, eff_frames)
        eff_steps = DEFAULT_INFERENCE_STEPS if num_inference_steps is None else int(num_inference_steps)
        eff_steps = max(15, min(150, eff_steps))
        eff_guidance = DEFAULT_GUIDANCE_SCALE if guidance_scale is None else float(guidance_scale)
        eff_guidance = max(1.001, min(30.0, eff_guidance))
        neg_used = DEFAULT_NEGATIVE_PROMPT if negative_prompt is None else negative_prompt

        logger.info(
            "ltx generate start | image=%s fps=%.3f frames=%d steps=%d guidance=%.2f seed=%s neg_len=%d prompt_len=%d",
            image_path,
            eff_fps,
            eff_frames,
            eff_steps,
            eff_guidance,
            str(seed),
            len(neg_used) if neg_used else 0,
            len(prompt),
        )
        started = time.perf_counter()
        pipe = self._load()

        base = Image.open(image_path).convert("RGB")
        raw_w, raw_h = base.size
        target_w, target_h = self._upscale_target_size(raw_w, raw_h)
        width = self._snap_to_32(target_w)
        height = self._snap_to_32(target_h)
        max_frames_env = os.getenv("LTX_MAX_FRAMES")
        hard_cap = int(max_frames_env) if max_frames_env else DEFAULT_MAX_FRAMES_CAP
        # Allow long clips when VRAM permits; override down with LTX_MAX_FRAMES if needed.
        hard_cap = max(9, min(1537, hard_cap))
        eff_frames_capped = min(eff_frames, hard_cap)
        num_frames = self._snap_frames(eff_frames_capped)
        if (width, height) != (raw_w, raw_h):
            base = base.resize((width, height), Image.BICUBIC)
        logger.info(
            "ltx normalized params | size=%dx%d->%dx%d frames=%d->%d model=%s",
            raw_w,
            raw_h,
            width,
            height,
            eff_frames,
            num_frames,
            self._active_model_id,
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        generator = None
        if DEVICE == "cuda":
            if seed is not None:
                generator = torch.Generator(device=DEVICE).manual_seed(int(seed) & 0x7FFFFFFF)
            else:
                generator = torch.Generator(device=DEVICE).manual_seed(int(time.time()) % 1_000_000)
        elif DEVICE == "cpu" and seed is not None:
            generator = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)

        safe_prompt = self._compress_prompt(prompt, max_words=60)
        if safe_prompt != prompt:
            logger.info("ltx prompt truncated | original_words=%d kept_words=%d", len(prompt.split()), len(safe_prompt.split()))

        call_kw: dict = {
            "image": base,
            "prompt": safe_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": eff_fps,
            "num_inference_steps": eff_steps,
            "guidance_scale": eff_guidance,
            "negative_prompt": neg_used,
            "generator": generator,
            "output_type": "np",
        }
        _variants = (
            call_kw,
            {k: v for k, v in call_kw.items() if k != "negative_prompt"},
            {k: v for k, v in call_kw.items() if k != "frame_rate"},
            {k: v for k, v in call_kw.items() if k not in ("frame_rate", "negative_prompt")},
        )
        last_type_err: Optional[TypeError] = None
        result = None
        for variant in _variants:
            filtered = _filter_pipeline_kwargs(pipe, variant)
            dropped = sorted(set(variant) - set(filtered))
            if dropped:
                logger.warning("ltx skipping unsupported kwargs | dropped=%s", dropped)
            try:
                result = pipe(**filtered)
                last_type_err = None
                break
            except TypeError as exc:
                last_type_err = exc
                logger.warning(
                    "ltx pipe TypeError — trying older diffusers kw set | omitting=%s | exc=%s",
                    sorted(set(call_kw.keys()) - set(variant.keys())),
                    exc,
                )
        if result is None:
            raise last_type_err if last_type_err else RuntimeError("ltx pipe returned empty result")

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

        encode_fps = max(8.0, min(120.0, float(eff_fps)))
        logger.info("ltx writing video | frames=%d encode_fps=%.3f", len(frames_np), encode_fps)
        writer = imageio.get_writer(
            str(output),
            fps=encode_fps,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=[
                "-movflags",
                "faststart",
                "-crf",
                "16",
                "-preset",
                "slow",
            ],
        )
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
