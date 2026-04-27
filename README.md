# 🎬 FLUX + LTX API — Image-to-Video Pipeline

RunPod RTX 4090 üzerinde çalışan **FLUX.1-dev** (image generation) ve **LTX-2.3-fp8** (image-to-video) API sunucusu.

Bu repo, [apexaic-film-otomasyon](https://github.com/Burak1320demiroz/apexaic-film-otomasyon) projesinin GPU backend'idir.

## 🚀 RunPod 4090 — Hızlı Kurulum

### 1) Pod Oluştur
- RunPod'da **RTX 4090** (24GB VRAM) pod oluştur
- Template: **RunPod PyTorch 2.x** (veya herhangi bir CUDA 12.x template)
- HTTP Port: `8000`

### 2) Clone & Setup (TEK KOMUT)

```bash
# Pod terminal'de:
git clone https://github.com/Burak1320demiroz/model-video-image.git /workspace/model-video-image
cd /workspace/model-video-image
bash scripts/runpod_setup.sh
```

> Setup script otomatik olarak:
> - GPU'yu tespit eder
> - PyTorch + CUDA 12.4 kurar
> - Proje bağımlılıklarını kurar
> - HuggingFace token'ını ayarlar

### 3) HuggingFace Login (İlk kez)

FLUX.1-dev **gated model**dir. Erişim için:

```bash
# https://huggingface.co/black-forest-labs/FLUX.1-dev adresinden erişim iste
huggingface-cli login
# Token'ınızı girin
```

### 4) Sunucuyu Başlat

```bash
./scripts/runpod_start.sh
```

Veya doğrudan:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5) Otomasyon ile Bağla

RunPod proxy URL'ini kopyala:
```
https://<pod-id>-8000.proxy.runpod.net
```

**apexaic-film-otomasyon** → Ayarlar → RunPod API URL alanına yapıştır → "Bağlantı Test Et" ile doğrula.

---

## 📡 API Endpointleri

### Health Check

```bash
GET /health
```

```json
{ "status": "ok" }
```

### Görsel Üretme (FLUX.1-dev)

```bash
POST /generate-image
```

```json
{
  "prompt": "cyberpunk city night, neon lights reflecting on wet streets",
  "project_name": "my-film"
}
```

Response:

```json
{
  "image_url": "output/my_film/img_ab12cd34.png",
  "image_public_url": "https://<pod-url>/output/my_film/img_ab12cd34.png",
  "image_download_url": "https://<pod-url>/download/my_film/img_ab12cd34.png"
}
```

### Video Üretme (LTX-2.3 Image-to-Video)

```bash
POST /generate-video
```

```json
{
  "image": "output/my_film/img_ab12cd34.png",
  "prompt": "camera slowly moves forward through the neon-lit street, rain droplets visible",
  "project_name": "my-film"
}
```

Response:

```json
{
  "video_url": "output/my_film/vid_ef56gh78.mp4",
  "video_public_url": "https://<pod-url>/output/my_film/vid_ef56gh78.mp4",
  "video_download_url": "https://<pod-url>/download/my_film/vid_ef56gh78.mp4"
}
```

### Dosya İndirme

```bash
GET /download/{project_name}/{file_name}
```

---

## 🔧 VRAM Yönetimi

API, 24GB VRAM'ı verimli kullanmak için otomatik model geçişi yapar:

- `/generate-image` çağrılınca **LTX modeli unload** edilir → FLUX yüklenir
- `/generate-video` çağrılınca **FLUX modeli unload** edilir → LTX yüklenir
- Aynı anda iki model asla bellekte tutulmaz
- `MODEL_LOCK` ile eşzamanlı istekler sıralanır

Varsayılan davranış: işlem bitince aktif model bellekte tutulur (hız için).  
`UNLOAD_AFTER_REQUEST=1` ile her istekten sonra unload yapılabilir.

---

## 🐳 Docker ile Deploy (Opsiyonel)

```bash
docker build -t your-repo/ai-server:latest .
docker push your-repo/ai-server:latest
```

RunPod'da Custom Container Image olarak kullan.

---

## 📋 Kontrol Listesi

| Kontrol | Komut |
|---------|-------|
| GPU görünüyor mu | `nvidia-smi` |
| PyTorch CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |
| API çalışıyor mu | `curl http://localhost:8000/health` |
| API docs | `http://localhost:8000/docs` |

---

## 🏗️ Mimari

```
Frontend (apexaic-film-otomasyon)
    │
    │  HTTP (RunPod proxy URL)
    ▼
┌─────────────────────────────┐
│  FastAPI Server (port 8000) │
│  ┌──────────┬──────────┐    │
│  │ FLUX.1   │ LTX-2.3  │    │
│  │ (image)  │ (video)  │    │
│  └──────────┴──────────┘    │
│       RTX 4090 (24GB)       │
└─────────────────────────────┘
```

**Pipeline Akışı:**
1. Otomasyon → AI prompt üretir (Groq LLM)
2. FLUX.1-dev → Sahne görseli üretir
3. LTX-2.3 → Görselden 8s video üretir
4. Video'lar birleştirilerek film oluşturulur
