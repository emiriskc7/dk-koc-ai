# ============================================================
# Koçluk RAG Chatbot — Tam Kurulum Scripti
# Python 3.13 + Windows
#
# Kullanım:
#   PowerShell'i projenin kök klasöründe aç ve çalıştır:
#     .\install.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$VenvDir = ".venv"
$PythonExe = "$VenvDir\Scripts\python.exe"
$PipExe   = "$VenvDir\Scripts\pip.exe"

function Log($msg) { Write-Host "`n[KURULUM] $msg" -ForegroundColor Cyan }
function Ok($msg)  { Write-Host "    OK: $msg" -ForegroundColor Green }

# --- Adım 1: Sanal ortam ---
Log "Adım 1 — Sanal ortam oluşturuluyor ($VenvDir)"
if (-not (Test-Path "$VenvDir\Scripts\python.exe")) {
    python -m venv $VenvDir
    Ok "Sanal ortam oluşturuldu."
} else {
    Ok "Zaten mevcut — atlanıyor."
}

# --- Adım 2: pip güncelle ---
Log "Adım 2 — pip güncelleniyor"
& $PythonExe -m pip install --upgrade pip --quiet
Ok "pip güncel."

# --- Adım 3: setuptools sabitle (torch 2.x, setuptools 82+ ile bozuluyor) ---
Log "Adım 3 — setuptools sürümü sabitleniyor (81.0.0)"
& $PipExe install "setuptools==81.0.0" --quiet
Ok "setuptools 81.0.0"

# --- Adım 4: PyTorch CPU (wheel sunucusundan; CUDA gerekmez) ---
Log "Adım 4 — PyTorch CPU kuruluyor"
& $PipExe install torch --index-url https://download.pytorch.org/whl/cpu --quiet
Ok "PyTorch CPU kuruldu."

# --- Adım 5: Ana bağımlılıklar ---
Log "Adım 5 — requirements.txt kuruluyor"
& $PipExe install -r requirements.txt --only-binary :all: --quiet
Ok "requirements.txt tamam."

# --- Adım 6: chromadb 1.x (Python 3.13 uyumlu) ---
Log "Adım 6 — chromadb 1.x kuruluyor"
# chromadb 0.5.x / 0.6.x = Python 3.13 için C++ Build Tools gerektirir.
# chromadb 1.x = pre-built wheel, derleme yok.
& $PipExe install "chromadb>=1.0.0" --only-binary :all: --quiet
Ok "chromadb 1.x kuruldu."

# --- Adım 7: langchain-chroma --no-deps ---
Log "Adım 7 — langchain-chroma 0.1.4 --no-deps ile kuruluyor"
# langchain-chroma 0.1.4 metadata'sında chromadb<0.6 yazıyor.
# Runtime'da chromadb 1.x ile sorunsuz çalışıyor.
# pip metadata çakışmasını --no-deps ile atlıyoruz.
& $PipExe install "langchain-chroma==0.1.4" --no-deps --quiet
Ok "langchain-chroma 0.1.4 kuruldu."

# --- Özet ---
Log "Kurulum tamamlandı!"
Write-Host ""
Write-Host "Çalıştırmak için:" -ForegroundColor Yellow
Write-Host "  $VenvDir\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  streamlit run app.py" -ForegroundColor White
Write-Host ""
