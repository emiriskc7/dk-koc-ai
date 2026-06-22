"""
config.py — Merkezi Yapılandırma Modülü
Tüm sabitler ve ortam değişkenleri buradan yönetilir.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================
# DIZIN YAPISI  (load_dotenv'den ÖNCE tanımlanmalı)
# =============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE_PATH = str(BASE_DIR / "coaching_knowledge_base.json")
VECTOR_STORE_PATH = str(BASE_DIR / "vector_store")

# =============================================================
# ORTAM DEĞİŞKENLERİNİ YÜKLE
# Önce .env, bulunamazsa .env.example'a bak (geliştirme kolaylığı)
# override=True: shell'deki eski değerlerin üzerine yazar
# =============================================================
_dot_env = BASE_DIR / ".env"
_dot_env_example = BASE_DIR / ".env.example"

if _dot_env.exists():
    load_dotenv(dotenv_path=_dot_env, override=True)
elif _dot_env_example.exists():
    load_dotenv(dotenv_path=_dot_env_example, override=True)
else:
    load_dotenv(override=True)  # son çare: sys.path üzerinde ara

# =============================================================
# EMBEDDING (Yerel — API maliyeti sıfır)
# =============================================================
# HuggingFace Model Hub'dan otomatik indirilir (~90 MB, ilk seferinde)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =============================================================
# VEKTÖR VERİTABANI (ChromaDB)
# =============================================================
COLLECTION_NAME = "coaching_kb"
TOP_K_RESULTS = 5          # retrieval'da döndürülecek doküman sayısı
MMR_FETCH_K = 20           # MMR için aday havuzu (çeşitlilik için)

# =============================================================
# VERİ İŞLEME (Chunking)
# =============================================================
CHUNK_SIZE = 800            # karakter cinsinden maksimum chunk boyutu
CHUNK_OVERLAP = 150         # chunk'lar arası örtüşme (bağlamı korur)

# =============================================================
# LLM — Google Gemini (Çift API Key Rotasyon Mimarisi)
# =============================================================
# Her iki anahtar .env dosyasından okunur.
# Key 1 kota/bağlantı hatası verirse with_fallbacks() Key 2'ye geçer.
GEMINI_API_KEY_1: str = os.getenv("GEMINI_API_KEY_1", "")
GEMINI_API_KEY_2: str = os.getenv("GEMINI_API_KEY_2", "")
GEMINI_MODEL_NAME: str = "gemini-3.1-flash-lite"  # Google API'nin aktif modeli

# =============================================================
# SOHBET BELLEK YÖNETİMİ
# =============================================================
MAX_CHAT_HISTORY: int = 100  # bellekte tutulacak maksimum mesaj çifti sayısı

# =============================================================
# UYGULAMA GENELİ
# =============================================================
APP_TITLE: str = "🎓 Derece Kampüsü Koçluk Asistanın"
APP_DESCRIPTION: str = (
    "Deneyimli koçlarımızın bilgi birikiminden güç alan YKS hazırlık asistanın."
)

# Koçların sohbet verilerinin bulunduğu JSON dosyasının yolu
# BASE_DIR ile birleştirilerek mutlak yol elde edilir (Streamlit hangi
# dizinden başlarsa başlasın dosyayı doğru bulur).
COACHES_CHAT_JSON_PATH: str = str(BASE_DIR / "data" / "coaching_knowledge_base.json")
COACH_PDF_DIR_PATH: str = str(BASE_DIR / "data" / "pdfs")
