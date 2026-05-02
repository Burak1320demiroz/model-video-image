#!/usr/bin/env bash
# ============================================================================
# RunPod 4090 — TEK KOMUTLA KURULUM
# ============================================================================
# Kullanım (RunPod Terminal'de):
#   git clone https://github.com/Burak1320demiroz/model-video-image.git /workspace/model-video-image
#   cd /workspace/model-video-image
#   bash scripts/runpod_setup.sh
#
# Bu script:
#   1. CUDA 12.4 uyumlu PyTorch kurar
#   2. Proje bağımlılıklarını kurar
#   3. HuggingFace token'ı ayarlar (varsa)
#   4. API sunucusunu başlatır (port 8000)
# ============================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-$(pwd)}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
PORT="${PORT:-8000}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   🎬 FLUX + LTX API — RunPod 4090 Setup                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ---- 1) GPU Kontrol ----
echo "🔍 [1/5] GPU kontrol ediliyor..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "   ✅ GPU: $GPU_NAME ($GPU_MEM MiB)"
else
    echo "   ⚠️  nvidia-smi bulunamadı — CPU modunda çalışacak"
fi

# ---- 2) pip güncelle ----
echo ""
echo "📦 [2/5] pip güncelleniyor..."
python -m pip install --upgrade pip -q

# ---- 3) PyTorch (CUDA 12.4+) ----
echo ""
echo "🔥 [3/5] PyTorch kontrol ediliyor..."
# Sadece eksikse veya eski sürümse kur, RunPod template'indeki PyTorch 2.8'i (RTX 5090 için) bozma
python -m pip install torch torchvision torchaudio -q

# PyTorch CUDA doğrulaması
CUDA_OK=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")
if [ "$CUDA_OK" = "yes" ]; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
    echo "   ✅ PyTorch $TORCH_VER (CUDA $CUDA_VER)"
else
    echo "   ⚠️  PyTorch CUDA desteği algılanmadı — devam ediliyor"
fi

# ---- 4) Proje bağımlılıkları ----
echo ""
echo "📋 [4/5] Proje bağımlılıkları kuruluyor..."
cd "$APP_DIR"
python -m pip install -r requirements.txt -q
echo "   ✅ requirements.txt kuruldu"

# ---- 5) HuggingFace token ----
echo ""
echo "🔑 [5/5] HuggingFace token ayarlanıyor..."
mkdir -p "$HF_HOME"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1

if [ -n "${HF_TOKEN:-}" ]; then
    echo "$HF_TOKEN" > "$HF_HOME/token"
    echo "   ✅ HF_TOKEN environment variable'dan aktarıldı"
elif [ -f "/root/.cache/huggingface/token" ]; then
    cp "/root/.cache/huggingface/token" "$HF_HOME/token"
    echo "   ✅ Token /root/.cache/huggingface/token 'dan kopyalandı"
elif [ -f "$HF_HOME/token" ]; then
    echo "   ✅ Token zaten $HF_HOME/token'da mevcut"
else
    echo "   ⚠️  HuggingFace token bulunamadı!"
    echo "      FLUX.1-dev gated model — aşağıdaki komutla giriş yapın:"
    echo "      huggingface-cli login"
    echo ""
fi

# ---- Özet ----
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   ✅ Kurulum tamamlandı!                                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║   Sunucuyu başlatmak için:                                 ║"
echo "║   ./scripts/runpod_start.sh                                ║"
echo "║                                                            ║"
echo "║   Veya doğrudan:                                           ║"
echo "║   python -m uvicorn main:app --host 0.0.0.0 --port $PORT  ║"
echo "║                                                            ║"
echo "║   API Docs: http://localhost:$PORT/docs                    ║"
echo "║   Health:   http://localhost:$PORT/health                  ║"
echo "║                                                            ║"
echo "║   RunPod proxy URL'ini otomasyon Settings'e girin:         ║"
echo "║   https://<pod-id>-$PORT.proxy.runpod.net                  ║"
echo "║                                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
