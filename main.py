import gc
import logging
import os
import re
import time
import uuid
from threading import Lock
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flux_api import FluxImageGenerator
from ltx_api import LtxVideoGenerator

# Default Hugging Face cache settings for RunPod.
# Uses /workspace to avoid filling the container overlay disk.
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/hf-cache/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


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
UNLOAD_AFTER_REQUEST = os.getenv("UNLOAD_AFTER_REQUEST", "0").lower() in {"1", "true", "yes"}


class ImageReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    seed: Optional[int] = None
    project_name: str = Field("default")
    scene_id: Optional[str] = None


class VideoReq(BaseModel):
    image: str = Field(..., min_length=1, description="Path like output/image.png")
    prompt: str = Field(..., min_length=1)
    project_name: str = Field("default")
    scene_id: Optional[str] = None

def get_safe_project_dir(base_dir: Path, proj_name: str) -> Path:
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', proj_name)
    if not safe_name:
        safe_name = "default"
    p = base_dir / safe_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_project_name(proj_name: str) -> str:
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', proj_name)
    return safe_name or "default"


def build_public_url(request: Request, rel_path: str) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{proto}://{host}/{rel_path}"


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
def generate_image(req: ImageReq, request: Request) -> dict:
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
            file_name = f"img_{req.scene_id}.png" if req.scene_id else f"img_{uuid.uuid4().hex[:8]}.png"
            
            image_path = flux.generate(
                prompt=req.prompt,
                output_path=str(p_dir / file_name),
                seed=req.seed,
            )
            # URL friendly path starting with output/
            rel_path = f"output/{safe_name}/{file_name}"
            download_url = build_public_url(request, f"download/{safe_name}/{file_name}")
            public_url = build_public_url(request, rel_path)
            
            elapsed = time.perf_counter() - started
            logger.info(
                "generate-image success | image=%s | took=%.2fs | %s",
                rel_path,
                elapsed,
                gpu_stats(),
            )
            return {
                "image_url": rel_path,
                "image_public_url": public_url,
                "image_download_url": download_url,
            }
        except Exception as exc:
            logger.exception("generate-image failed | %s", gpu_stats())
            raise HTTPException(status_code=500, detail=f"image generation failed: {exc}") from exc
        finally:
            if UNLOAD_AFTER_REQUEST:
                logger.info("generate-image finally unload flux")
                flux.unload()
                release_vram()
            else:
                logger.info("generate-image keeping flux loaded for faster next request")
            logger.info("generate-image cleanup complete | %s", gpu_stats())


@app.post("/generate-video")
def generate_video(req: VideoReq, request: Request) -> dict:
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
            file_name = f"vid_{req.scene_id}.mp4" if req.scene_id else f"vid_{uuid.uuid4().hex[:8]}.mp4"
            
            video_path = ltx.generate(
                image_path=str(image_path),
                prompt=req.prompt,
                output_path=str(p_dir / file_name),
            )
            rel_path = f"output/{safe_name}/{file_name}"
            download_url = build_public_url(request, f"download/{safe_name}/{file_name}")
            public_url = build_public_url(request, rel_path)
            
            elapsed = time.perf_counter() - started
            logger.info(
                "generate-video success | video=%s | took=%.2fs | %s",
                rel_path,
                elapsed,
                gpu_stats(),
            )
            return {
                "video_url": rel_path,
                "video_public_url": public_url,
                "video_download_url": download_url,
            }
        except Exception as exc:
            logger.exception("generate-video failed | %s", gpu_stats())
            raise HTTPException(status_code=500, detail=f"video generation failed: {exc}") from exc
        finally:
            if UNLOAD_AFTER_REQUEST:
                logger.info("generate-video finally unload ltx")
                ltx.unload()
                release_vram()
            else:
                logger.info("generate-video keeping ltx loaded for faster next request")
            logger.info("generate-video cleanup complete | %s", gpu_stats())


@app.get("/download/{project_name}/{file_name}")
def download_file(
    project_name: str,
    file_name: str,
    download: bool = Query(True, description="Serve as attachment when true"),
) -> FileResponse:
    safe_proj = safe_project_name(project_name)
    requested_path = (output_dir / safe_proj / file_name).resolve()
    output_root = output_dir.resolve()

    if output_root not in requested_path.parents:
        raise HTTPException(status_code=400, detail="invalid download path")
    if not requested_path.exists() or not requested_path.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    disposition_filename = requested_path.name if download else None
    media_type = "application/octet-stream" if download else None
    return FileResponse(
        path=str(requested_path),
        filename=disposition_filename,
        media_type=media_type,
    )


@app.get("/list-files")
def list_files(project_name: str, request: Request) -> dict:
    safe_proj = safe_project_name(project_name)
    proj_dir = (output_dir / safe_proj).resolve()
    
    files_dict = {}
    if proj_dir.exists() and proj_dir.is_dir():
        for f in proj_dir.iterdir():
            if f.is_file():
                rel_path = f"output/{safe_proj}/{f.name}"
                download_path = f"download/{safe_proj}/{f.name}"
                files_dict[f.name] = {
                    "rel_path": rel_path,
                    "download_url": build_public_url(request, download_path)
                }
                
    return {"project_name": project_name, "files": files_dict}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
