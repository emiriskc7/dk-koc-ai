"""
config.py — Merkezi Yapılandırma Modülü
Tüm sabitler ve ortam değişkenleri buradan yönetilir.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE_PATH = str(BASE_DIR / "coaching_knowledge_base.json")
VECTOR_STORE_PATH = str(BASE_DIR / "vector_store")

_dot_env = BASE_DIR / ".env"
_dot_env_example = BASE_DIR / ".env.example"

if _dot_env.exists():
    load_dotenv(dotenv_path=_dot_env, override=True)
elif _dot_env_example.exists():
    load_dotenv(dotenv_path=_dot_env_example, override=True)
else:
    load_dotenv(override=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "coaching_kb"
TOP_K_RESULTS = 5
MMR_FETCH_K = 20

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

GEMINI_API_KEY_1: str = os.getenv("GEMINI_API_KEY_1", "")
GEMINI_API_KEY_2: str = os.getenv("GEMINI_API_KEY_2", "")
GEMINI_MODEL_NAME: str = "gemini-3.1-flash-lite"

MAX_CHAT_HISTORY: int = 100

APP_TITLE: str = "🎓 Derece Kampüsü Koçluk Asistanın"
APP_DESCRIPTION: str = (
    "Deneyimli koçlarımızın bilgi birikiminden güç alan YKS hazırlık asistanın."
)

COACHES_CHAT_JSON_PATH: str = str(BASE_DIR / "data" / "coaching_knowledge_base.json")
COACH_PDF_DIR_PATH: str = str(BASE_DIR / "data" / "pdfs")
