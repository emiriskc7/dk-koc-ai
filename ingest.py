#!/usr/bin/env python3
"""
ingest.py — Veri Hazırlama ve Vektörleştirme Scripti
=====================================================
Bu script tek seferlik (veya veri güncellendiğinde) çalıştırılır.

İşlem adımları:
  1. JSON bilgi tabanını yükle ve temizle
  2. Her Q&A çiftini zengin metadata ile LangChain Document'a dönüştür
  3. RecursiveCharacterTextSplitter ile chunk'lara böl
  4. Yerel HuggingFace embedding ile vektörleştir (API maliyeti = 0)
  5. ChromaDB'ye diske kaydet (her başlatmada yeniden işlenmez)

Kullanım:
  python ingest.py
  python ingest.py --reset   # Mevcut vektör mağazasını silip yeniden oluşturur
"""

import json
import logging
import shutil
import sys
import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Proje kök dizinini sys.path'e ekle (config.py'ye erişim için)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_FILE_PATH,
    VECTOR_STORE_PATH,
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# =============================================================
# LOGLAMA
# =============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Filtre: WhatsApp sistem mesajları ve anlamsız kısa metinler
_NOISE_PATTERNS = [
    "topluluk üzerinden katıldı",
    "gruba hoş geldiniz",
    "artık yöneticisiniz",
    "bu mesaj silindi",
    "şifrelidir",
    "yalnızca yöneticiler",
    "grubuna katıldı",
    "ayrıldı",
    "kişisini ekledi",
    "sohbet grubunu oluşturdunuz",
    "ayarlarını değiştirdi",
]


# =============================================================
# YARDIMCI FONKSİYONLAR
# =============================================================

def _is_noise(text: str) -> bool:
    """WhatsApp sistem mesajı veya anlamsız kısa metni tespit eder."""
    if not text or len(text.strip()) < 10:
        return True
    lower = text.lower()
    return any(pattern in lower for pattern in _NOISE_PATTERNS)


def _clean_name(name: str) -> str:
    """Koç isimlerindeki '~' ve boşlukları temizler."""
    return name.lstrip("~").strip() if name else "Bilinmiyor"


# =============================================================
# ADIM 1 — JSON'u Document listesine çevir
# =============================================================

def load_json_as_documents(file_path: str) -> List[Document]:
    """
    coaching_knowledge_base.json formatını okur.
    Beklenen format:
      [
        {
          "category": "Genel",
          "question": "...",
          "asked_by": "~Koç İsmi",
          "possible_answers": [
            {"answer": "...", "answered_by": "~Koç İsmi"},
            ...
          ]
        },
        ...
      ]

    Her geçerli Q&A çifti tek bir Document'a dönüştürülür.
    Gürültülü / sistem mesajları filtrelenir.
    """
    logger.info(f"JSON bilgi tabanı yükleniyor: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON dosyası en üst düzeyde bir liste (array) olmalıdır.")

    documents: List[Document] = []
    skipped = 0

    for idx, item in enumerate(tqdm(data, desc="Dokümanlar işleniyor", unit="kayıt")):
        category = item.get("category", "Genel").strip()
        question = (item.get("question") or "").strip()
        asked_by = _clean_name(item.get("asked_by", ""))
        raw_answers = item.get("possible_answers") or []

        # Gürültülü veya anlamsız soruları atla
        if _is_noise(question):
            skipped += 1
            continue

        # Geçerli cevapları filtrele ve birleştir
        valid_answers = []
        for ans in raw_answers:
            answer_text = (ans.get("answer") or "").strip()
            answered_by = _clean_name(ans.get("answered_by", ""))
            if not _is_noise(answer_text):
                valid_answers.append(f"  • {answered_by}: {answer_text}")

        # Cevapsız kayıtları da dahil et (soruyu soranı bilgi olarak sakla)
        answers_block = (
            "\n".join(valid_answers)
            if valid_answers
            else "  • (Bu soruya henüz cevap verilmemiş)"
        )

        page_content = (
            f"[KATEGORİ]: {category}\n"
            f"[SORU]: {question}\n"
            f"[SORAN KOÇ]: {asked_by}\n"
            f"[KOÇLARIN CEVAPLARI]:\n{answers_block}"
        )

        doc = Document(
            page_content=page_content,
            metadata={
                "doc_idx": idx,
                "category": category,
                "asked_by": asked_by,
                # Metadata arama için kısa versiyon (ChromaDB 512 char sınırı)
                "question_short": question[:300],
                "answer_count": len(valid_answers),
            },
        )
        documents.append(doc)

    logger.info(
        f"Yükleme tamamlandı: {len(documents)} geçerli doküman, "
        f"{skipped} gürültülü kayıt atlandı."
    )
    return documents


# =============================================================
# ADIM 2 — Chunk'lama
# =============================================================

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Dokümanları semantic olarak anlamlı chunk'lara böler.
    Q&A çiftleri genellikle kısadır; çoğu bölünmeden geçer.
    CHUNK_SIZE aşıldığında devreye girer.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # Önce başlık bloklarından böl, mecbur kalınca kelimeden
        separators=["\n[", "\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        f"Chunk'lama tamamlandı: {len(documents)} doküman → {len(chunks)} chunk "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


# =============================================================
# ADIM 3 — Embed et ve ChromaDB'ye kaydet
# =============================================================

def build_vector_store(chunks: List[Document], reset: bool = False) -> Chroma:
    """
    Yerel HuggingFace embedding kullanarak chunk'ları vektörleştirir
    ve ChromaDB'ye diske kalıcı olarak kaydeder.

    - İlk çalıştırmada embedding modeli (~90 MB) indirilir.
    - Sonraki çalıştırmalarda diskten okunur (yeniden embed EDİLMEZ).
    - reset=True ise mevcut mağaza silinip yeniden oluşturulur.
    """
    store_path = Path(VECTOR_STORE_PATH)

    if store_path.exists():
        if reset:
            logger.info(f"Eski vektör mağazası siliniyor: {store_path}")
            shutil.rmtree(store_path)
        else:
            logger.warning(
                f"Vektör mağazası zaten mevcut: {store_path}\n"
                "Yeniden oluşturmak için '--reset' bayrağını kullanın.\n"
                "Mevcut mağaza korunuyor, çıkılıyor..."
            )
            return _load_existing_store()

    logger.info(f"Embedding modeli yükleniyor: {EMBEDDING_MODEL_NAME}")
    logger.info("(İlk çalıştırmada model ~90 MB indirilir, lütfen bekleyin...)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(f"Vektörleştirme başlıyor: {len(chunks)} chunk işlenecek...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(store_path),
        collection_name=COLLECTION_NAME,
    )

    doc_count = vectorstore._collection.count()
    logger.info(
        f"ChromaDB oluşturuldu ve diske kaydedildi: {store_path}\n"
        f"Toplam vektör sayısı: {doc_count}"
    )
    return vectorstore


def _load_existing_store() -> Chroma:
    """Diskten mevcut ChromaDB mağazasını yükler (embed etmeden)."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# =============================================================
# MAIN
# =============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Koçluk bilgi tabanını vektörleştirir ve ChromaDB'ye kaydeder."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Mevcut vektör mağazasını silip sıfırdan oluşturur.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_FILE_PATH,
        help=f"JSON veri dosyası yolu (varsayılan: {DATA_FILE_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data

    # Veri dosyası kontrolü
    if not Path(data_path).exists():
        logger.error(f"Veri dosyası bulunamadı: {data_path}")
        logger.error("Lütfen coaching_knowledge_base.json dosyasının proje kökünde olduğundan emin olun.")
        sys.exit(1)

    logger.info("=" * 55)
    logger.info("  KOÇLUK CHATBOT — VERİ İNGESTION BAŞLIYOR")
    logger.info("=" * 55)

    # Adım 1: Yükle ve temizle
    documents = load_json_as_documents(data_path)

    if not documents:
        logger.error("Hiç geçerli doküman yüklenemedi. JSON formatını kontrol edin.")
        sys.exit(1)

    # Adım 2: Chunk'la
    chunks = split_documents(documents)

    # Adım 3: Vektörleştir ve kaydet
    build_vector_store(chunks, reset=args.reset)

    logger.info("=" * 55)
    logger.info("  INGESTION TAMAMLANDI!")
    logger.info("  Sonraki adım: streamlit run app.py")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
