import gc
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class LtxVideoGenerator:
    """
    Placeholder LTX-style video generator.
    Keeps a lazy-load interface so you can replace internals with a real LTX model.
    """

    def __init__(self, model_id: str = "Lightricks/LTX-Video") -> None:
        self.model_id = model_id
        self._loaded = False

    def _load(self) -> None:
        if not self._loaded:
            # Real model load can be inserted here.
            self._loaded = True

    def unload(self) -> None:
        self._loaded = False
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    def generate(
        self,
        image_path: str,
        prompt: str,
        output_path: str = "output/video.mp4",
        fps: int = 12,
        frames: int = 48,
    ) -> str:
        """
        Creates a simple animated video from a source image.
        This is a production-safe fallback until real LTX weights are wired.
        """
        _ = prompt
        self._load()

        base = Image.open(image_path).convert("RGB")
        arr = np.asarray(base, dtype=np.float32)
        h, w, _ = arr.shape

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        writer = imageio.get_writer(str(output), fps=fps, codec="libx264")
        try:
            for i in range(frames):
                t = i / max(frames - 1, 1)
                zoom = 1.0 + (0.05 * t)
                nh, nw = int(h / zoom), int(w / zoom)
                y0 = (h - nh) // 2
                x0 = (w - nw) // 2
                crop = arr[y0 : y0 + nh, x0 : x0 + nw]
                frame = Image.fromarray(crop.astype(np.uint8)).resize((w, h), Image.BICUBIC)
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()

        return str(output)
