import gc
import logging
import re
import time
import uuid
from threading import Lock
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flux_api import FluxImageGenerator
from ltx_api import LtxVideoGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(title="FLUX + LTX API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

flux = FluxImageGenerator()
ltx = LtxVideoGenerator()
MODEL_LOCK = Lock()


class ImageReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    seed: Optional[int] = None
    project_name: str = Field("default")


class VideoReq(BaseModel):
    image: str = Field(..., min_length=1, description="Path like output/image.png")
    prompt: str = Field(..., min_length=1)
    project_name: str = Field("default")

def get_safe_project_dir(base_dir: Path, proj_name: str) -> Path:
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', proj_name)
    if not safe_name:
        safe_name = "default"
    p = base_dir / safe_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def release_vram() -> None:
    logger.info("releasing vram cache")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_stats() -> str:
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    return (
        f"gpu={torch.cuda.get_device_name(0)} "
        f"alloc={allocated:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB"
    )


@app.on_event("startup")
def on_startup() -> None:
    logger.info("api startup | %s", gpu_stats())


@app.get("/health")
def health() -> dict:
    logger.info("health check | %s", gpu_stats())
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {"service": "FLUX + LTX API", "status": "ok"}


@app.get("/api/status")
def api_status() -> dict:
    return {"status": "ok"}


@app.post("/generate-image")
def generate_image(req: ImageReq) -> dict:
    started = time.perf_counter()
    logger.info("generate-image requested | seed=%s | prompt_len=%d", req.seed, len(req.prompt))
    with MODEL_LOCK:
        logger.info("generate-image lock acquired")
        # Never keep both models in memory at once.
        logger.info("generate-image unloading ltx before flux | %s", gpu_stats())
        ltx.unload()
        release_vram()
        try:
            logger.info("generate-image flux.generate start | %s", gpu_stats())
            
            p_dir = get_safe_project_dir(output_dir, req.project_name)
            safe_name = p_dir.name
            file_name = f"img_{uuid.uuid4().hex[:8]}.png"
            
            image_path = flux.generate(
                prompt=req.prompt,
                output_path=str(p_dir / file_name),
                seed=req.seed,
            )
            # URL friendly path starting with output/
            rel_path = f"output/{safe_name}/{file_name}"
            
            elapsed = time.perf_counter() - started
            logger.info(
                "generate-image success | image=%s | took=%.2fs | %s",
                rel_path,
                elapsed,
                gpu_stats(),
            )
            return {"image_url": rel_path}
        except Exception as exc:
            logger.exception("generate-image failed | %s", gpu_stats())
            raise HTTPException(status_code=500, detail=f"image generation failed: {exc}") from exc
        finally:
            logger.info("generate-image finally unload flux")
            flux.unload()
            release_vram()
            logger.info("generate-image cleanup complete | %s", gpu_stats())


@app.post("/generate-video")
def generate_video(req: VideoReq) -> dict:
    started = time.perf_counter()
    logger.info("generate-video requested | image=%s | prompt_len=%d", req.image, len(req.prompt))
    image_path = Path(req.image)
    if not image_path.exists():
        logger.warning("generate-video image not found | image=%s", req.image)
        raise HTTPException(status_code=400, detail=f"image not found: {req.image}")

    with MODEL_LOCK:
        logger.info("generate-video lock acquired")
        # Never keep both models in memory at once.
        logger.info("generate-video unloading flux before ltx | %s", gpu_stats())
        flux.unload()
        release_vram()
        try:
            logger.info("generate-video ltx.generate start | %s", gpu_stats())
            
            p_dir = get_safe_project_dir(output_dir, req.project_name)
            safe_name = p_dir.name
            file_name = f"vid_{uuid.uuid4().hex[:8]}.mp4"
            
            video_path = ltx.generate(
                image_path=str(image_path),
                prompt=req.prompt,
                output_path=str(p_dir / file_name),
            )
            rel_path = f"output/{safe_name}/{file_name}"
            
            elapsed = time.perf_counter() - started
            logger.info(
                "generate-video success | video=%s | took=%.2fs | %s",
                rel_path,
                elapsed,
                gpu_stats(),
            )
            return {"video_url": rel_path}
        except Exception as exc:
            logger.exception("generate-video failed | %s", gpu_stats())
            raise HTTPException(status_code=500, detail=f"video generation failed: {exc}") from exc
        finally:
            logger.info("generate-video finally unload ltx")
            ltx.unload()
            release_vram()
            logger.info("generate-video cleanup complete | %s", gpu_stats())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
