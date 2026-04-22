import gc
from threading import Lock
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flux_api import FluxImageGenerator
from ltx_api import LtxVideoGenerator


app = FastAPI(title="FLUX + LTX API", version="1.0.0")

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

flux = FluxImageGenerator()
ltx = LtxVideoGenerator()
MODEL_LOCK = Lock()


class ImageReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    seed: Optional[int] = None


class VideoReq(BaseModel):
    image: str = Field(..., min_length=1, description="Path like output/image.png")
    prompt: str = Field(..., min_length=1)


def release_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate-image")
def generate_image(req: ImageReq) -> dict:
    with MODEL_LOCK:
        # Never keep both models in memory at once.
        ltx.unload()
        release_vram()
        try:
            image_path = flux.generate(
                prompt=req.prompt,
                output_path="output/image.png",
                seed=req.seed,
            )
            return {"image_url": image_path}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"image generation failed: {exc}") from exc
        finally:
            flux.unload()
            release_vram()


@app.post("/generate-video")
def generate_video(req: VideoReq) -> dict:
    image_path = Path(req.image)
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"image not found: {req.image}")

    with MODEL_LOCK:
        # Never keep both models in memory at once.
        flux.unload()
        release_vram()
        try:
            video_path = ltx.generate(
                image_path=str(image_path),
                prompt=req.prompt,
                output_path="output/video.mp4",
            )
            return {"video_url": video_path}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"video generation failed: {exc}") from exc
        finally:
            ltx.unload()
            release_vram()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
