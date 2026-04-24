# FLUX + LTX API (RunPod)

Bu proje, tek Docker container icinde iki endpoint ile calisir:

- `POST /generate-image`
- `POST /generate-video`
- `GET /health`

> Not: Gercek kullanim hedefi RunPod RTX 4090'dur.

## 1) Deploy Ozeti (RunPod)
<!--  -->
1. Image olustur:
   - `docker build -t seninrepo/ai-server:latest .`
2. Docker Hub'a push et:
   - `docker push seninrepo/ai-server:latest`
3. RunPod'da Pod olustur:
   - Custom Container Image: `seninrepo/ai-server:latest`
   - GPU: RTX 4090
   - HTTP Port: `8000`
4. Pod URL'ini al:
   - Ornek: `https://abcd1234-8000.proxy.runpod.net`

## 2) Endpointler

### Health Check

- `GET /health`

Ornek cevap:

```json
{
  "status": "ok"
}
```

### Gorsel Uretme

- `POST /generate-image`

Request:

```json
{
  "prompt": "cyberpunk city night"
}
```

Response:

```json
{
  "image_url": "output/image.png"
}
```

### Video Uretme

- `POST /generate-video`

Request:

```json
{
  "image": "output/image.png",
  "prompt": "camera slowly moving forward"
}
```

Response:

```json
{
  "video_url": "output/video.mp4"
}
```

## 3) cURL ile istek atma

Asagida `POD_URL` degiskenini kendi RunPod URL'in ile degistir.

```bash
POD_URL="https://abcd1234-8000.proxy.runpod.net"
```

### 3.1 Health

```bash
curl -X GET "$POD_URL/health"
```

### 3.2 Image Generate

```bash
curl -X POST "$POD_URL/generate-image" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"cyberpunk city night"}'
```

### 3.3 Video Generate

```bash
curl -X POST "$POD_URL/generate-video" \
  -H "Content-Type: application/json" \
  -d '{"image":"output/image.png","prompt":"camera slowly moving forward"}'
```

## 4) Python (requests) ile istek atma

```python
import requests

POD_URL = "https://abcd1234-8000.proxy.runpod.net"

# 1) Image
r1 = requests.post(
    f"{POD_URL}/generate-image",
    json={"prompt": "cyberpunk city night"},
    timeout=600,
)
r1.raise_for_status()
image_data = r1.json()
print("Image response:", image_data)

# 2) Video
r2 = requests.post(
    f"{POD_URL}/generate-video",
    json={
        "image": image_data["image_url"],
        "prompt": "camera slowly moving forward",
    },
    timeout=600,
)
r2.raise_for_status()
video_data = r2.json()
print("Video response:", video_data)
```

## 5) Uretilen dosyalara erisim

API `output` klasorunu static olarak sunar.

Ornek:

- Gorsel: `https://<pod-url>/output/image.png`
- Video: `https://<pod-url>/output/video.mp4`

`generate-image` ve `generate-video` cevaplarinda gelen path'i pod URL ile birlestirerek direkt dosyayi indirebilirsin.

## 6) Onemli Notlar

- Kritik kullanim tavsiyesi:
  - Ayni anda FLUX ve LTX modelini yukleme.
  - Dogru sira:
    1. FLUX load -> image uret
    2. modeli RAM/VRAM'den sil
    3. LTX load -> video uret
- API bu repoda artik bu sirayi kod seviyesinde zorunlu uygular:
  - `/generate-image` cagrisi once LTX'i unload eder, is bitince FLUX'u unload eder.
  - `/generate-video` cagrisi once FLUX'u unload eder, is bitince LTX'i unload eder.
  - Endpointler tek bir model kilidi ile sirali calisir; ayni anda iki modelin bellekte tutulmasi engellenir.
- Islem sirasi onerisi:
  1. `generate-image`
  2. VRAM temizligi
  3. `generate-video`
- Bu siralama VRAM baskisini azaltir.
- Ilk istek, model yukleme nedeniyle daha yavas olabilir.
- Video modeli secimi:
  - Varsayilan: `Lightricks/LTX-Video` (RunPod 3090 icin stabil secim)
  - Istege gore `LTX-2.3` ailesi ayri ortamda denenebilir.
  - Promptlar model limitine gore otomatik kisaltilir (video tarafi).

## 7) RunPod Temiz Kurulum (Onerilen)

Asagidaki adimlar, CUDA/PyTorch uyumsuzlugu yasamadan temiz kurulum icindir.

1. Projeye gir:
   - `cd /workspace/model-video-image`
2. RunPod'un GPU uyumlu PyTorch surumlerini dogrudan kur:
   - `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124`
3. Proje bagimliliklarini kur:
   - `pip install -r requirements.txt`
4. Hugging Face cache'i `/workspace` altina al (disk dolmasini onler):
   - `mkdir -p /workspace/hf-cache/{hub,transformers}`
   - `export HF_HOME=/workspace/hf-cache`
   - `export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub`
   - `export HF_HUB_DISABLE_XET=1`
5. API'yi baslat:
   - `HF_HOME=/workspace/hf-cache HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub HF_HUB_DISABLE_XET=1 uvicorn main:app --host 0.0.0.0 --port 8000`

Kontroller:

- GPU gorunuyor mu:
  - `nvidia-smi`
- PyTorch CUDA goruyor mu:
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"`
- API dokumani:
  - `https://<pod-url>/docs`

Notlar:

- Bu repoda `torch` artik `requirements.txt` icinde tutulmuyor; RunPod taban imaji ile uyumlu surumu manuel kurmaniz gerekir.
- Ana sayfa (`/`) tanimli degildir, bu nedenle `/` icin `404` normaldir.
- `black-forest-labs/FLUX.2-dev` gated modeldir; Hugging Face hesabinizda model erisimi ve token login'i gerekli.
