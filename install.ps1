
$ErrorActionPreference = "Stop"
$VenvDir = ".venv"
$PythonExe = "$VenvDir\Scripts\python.exe"
$PipExe   = "$VenvDir\Scripts\pip.exe"

function Log($msg) { Write-Host "`n[KURULUM] $msg" -ForegroundColor Cyan }
function Ok($msg)  { Write-Host "    OK: $msg" -ForegroundColor Green }


Log "Adım 1 — Sanal ortam oluşturuluyor ($VenvDir)"
if (-not (Test-Path "$VenvDir\Scripts\python.exe")) {
    python -m venv $VenvDir
    Ok "Sanal ortam oluşturuldu."
} else {
    Ok "Zaten mevcut — atlanıyor."
}


Log "Adım 2 — pip güncelleniyor"
& $PythonExe -m pip install --upgrade pip --quiet
Ok "pip güncel."


Log "Adım 3 — setuptools sürümü sabitleniyor (81.0.0)"
& $PipExe install "setuptools==81.0.0" --quiet
Ok "setuptools 81.0.0"


Log "Adım 4 — PyTorch CPU kuruluyor"
& $PipExe install torch --index-url https://download.pytorch.org/whl/cpu --quiet
Ok "PyTorch CPU kuruldu."


Log "Adım 5 — requirements.txt kuruluyor"
& $PipExe install -r requirements.txt --only-binary :all: --quiet
Ok "requirements.txt tamam."


Log "Adım 6 — chromadb 1.x kuruluyor"

& $PipExe install "chromadb>=1.0.0" --only-binary :all: --quiet
Ok "chromadb 1.x kuruldu."


Log "Adım 7 — langchain-chroma 0.1.4 --no-deps ile kuruluyor"

& $PipExe install "langchain-chroma==0.1.4" --no-deps --quiet
Ok "langchain-chroma 0.1.4 kuruldu."

Log "Kurulum tamamlandı!"
Write-Host ""
Write-Host "Çalıştırmak için:" -ForegroundColor Yellow
Write-Host "  $VenvDir\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  streamlit run app.py" -ForegroundColor White
Write-Host ""
