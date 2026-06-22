"""
retriever.py — Vektör Mağazası Erişim Katmanı
===============================================
ChromaDB'yi diskten bir kez yükler (lru_cache) ve
benzerlik eşiği (score_threshold) uygulayan bir retriever döner.

Score Threshold mantığı:
  - Cosine benzerlik skoru eşiğin ALTINDA kalan dokümanlar
    filtrelenir ve LLM'e GÖNDERİLMEZ.
  - Bu, alakasız içeriklerin modele ulaşmasını ve halüsinasyonu
    önlemenin en etkili katman-bazlı yoludur.

Eşik Ayarı (SCORE_THRESHOLD):
  0.30 → Gevşek  (alakasız sonuçlar gelebilir)
  0.40 → Dengeli ← varsayılan
  0.55 → Sıkı    (az ama çok isabetli sonuçlar)
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from config import (
    VECTOR_STORE_PATH,
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
    TOP_K_RESULTS,
)

logger = logging.getLogger(__name__)

SCORE_THRESHOLD: float = 0.40


# ─────────────────────────────────────────────────────────────
# EMBEDDING & VEKTÖR MAĞAZASI (Tek Örnekli — lru_cache)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    HuggingFace embedding modelini bir kez yükler ve önbelleğe alır.
    İlk çağrıda model indirme (~90 MB) gerçekleşebilir.
    """
    logger.info(f"Embedding modeli yükleniyor: {EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    """
    ChromaDB'yi diskten yükler; mevcut değilse kullanıcıyı bilgilendiren
    açıklayıcı bir hata fırlatır.
    """
    store_path = Path(VECTOR_STORE_PATH)

    if not store_path.exists():
        raise FileNotFoundError(
            f"\n{'='*50}\n"
            f"Vektör mağazası bulunamadı: {store_path}\n"
            f"Lütfen önce aşağıdaki komutu çalıştırın:\n"
            f"  python ingest.py\n"
            f"{'='*50}"
        )

    embeddings = _get_embeddings()
    logger.info(f"ChromaDB diskten yükleniyor: {store_path}")

    vectorstore = Chroma(
        persist_directory=str(store_path),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    doc_count = vectorstore._collection.count()
    logger.info(f"ChromaDB hazır — {doc_count} vektör yüklendi.")
    return vectorstore


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def get_retriever(
    score_threshold: Optional[float] = None,
) -> VectorStoreRetriever:
    """
    Benzerlik eşiği uygulayan LangChain retriever döner.

    Parameters
    ----------
    score_threshold : Cosine benzerlik eşiği (0.0–1.0).
                      None → SCORE_THRESHOLD sabitini kullan.

    Returns
    -------
    VectorStoreRetriever — LangChain chain'lerine bağlanabilir retriever.
    """
    threshold = score_threshold if score_threshold is not None else SCORE_THRESHOLD
    vectorstore = _get_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": TOP_K_RESULTS,
            "score_threshold": threshold,
        },
    )


def get_vectorstore_stats() -> dict:
    """
    Sidebar için vektör mağazası bilgilerini döner.
    Hata durumunda güvenli varsayılan değerler verir (uygulamayı çökertmez).
    """
    try:
        vs = _get_vectorstore()
        return {
            "status": "ok",
            "doc_count": vs._collection.count(),
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME.split("/")[-1],
            "score_threshold": SCORE_THRESHOLD,
        }
    except FileNotFoundError:
        return {
            "status": "not_found",
            "doc_count": 0,
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME.split("/")[-1],
            "score_threshold": SCORE_THRESHOLD,
        }
    except Exception as exc:
        logger.error(f"Vektör mağazası istatistik hatası: {exc}")
        return {
            "status": "error",
            "doc_count": 0,
            "collection": "-",
            "embedding_model": "-",
            "score_threshold": SCORE_THRESHOLD,
        }
