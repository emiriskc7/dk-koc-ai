"""
llm_factory.py — Çift Gemini API Key Rotasyon Mimarisi
=======================================================
Mimari:
  Key 1 (GEMINI_API_KEY_1) → Ana model
  Key 2 (GEMINI_API_KEY_2) → Yedek model (şelale)

  Key 1 kota (Rate Limit) veya bağlantı hatası verirse,
  LangChain'in with_fallbacks() mekanizması hissettirmeden
  Key 2'ye otomatik geçiş yapar.

  Model: gemini-3.1-flash-lite (temperature=0.1, sıfır halüsinasyon)

Terminal Loglama:
  Key 1 → Key 2 geçişi terminale "⚠️ Gemini Key 1 limit dışı,
  Key 2'ye sorunsuz geçiş yapıldı." logu düşer.
"""

import logging
from functools import lru_cache
from typing import Optional, List, Dict, Any, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

from config import GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_MODEL_NAME

logger = logging.getLogger(__name__)


class _GeminiKeyLogger(BaseCallbackHandler):
    """
    Her Gemini key için terminal logu üretir.
    Kullanıcı arayüzüne (Streamlit) hiçbir şey yansımaz.
    """

    def __init__(self, label: str) -> None:
        super().__init__()
        self._label = label

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        print(f"\n[🔑 GEMINI] ▶  {self._label} devreye girdi")
        logger.info("[GEMINI] %s çağrılıyor", self._label)

    def on_llm_error(self, error, **kwargs) -> None:
        if self._label == "Gemini Key 1":
            print("⚠️ Gemini Key 1 limit dışı, Key 2'ye sorunsuz geçiş yapıldı.")
        else:
            print(f"[🔑 GEMINI] ✗  {self._label} hata verdi → {error}")
        logger.warning("[GEMINI] %s hata verdi: %s", self._label, error)


@lru_cache(maxsize=1)
def create_fallback_llm(temperature: float = 0.1) -> BaseChatModel:
    """
    İki Gemini API anahtarıyla çift-katmanlı şelale LLM kurar.

    Key 1 kota/bağlantı hatası verirse Key 2'ye otomatik geçilir.
    Her iki anahtar da yoksa EnvironmentError fırlatır.
    temperature=0.1 → Sıfır halüsinasyon için düşük yaratıcılık.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not GEMINI_API_KEY_1 and not GEMINI_API_KEY_2:
        raise EnvironmentError(
            "Hiçbir Gemini API anahtarı bulunamadı!\n"
            ".env dosyasına GEMINI_API_KEY_1 ve/veya GEMINI_API_KEY_2 ekleyin.\n"
            "Örnek: GEMINI_API_KEY_1=AIza..."
        )

    available: list[tuple[str, BaseChatModel]] = []

    if GEMINI_API_KEY_1:
        llm1 = ChatGoogleGenerativeAI(
            google_api_key=GEMINI_API_KEY_1,
            model=GEMINI_MODEL_NAME,
            temperature=temperature,
            max_retries=0,
            callbacks=[_GeminiKeyLogger("Gemini Key 1")],
        )
        available.append(("Gemini Key 1", llm1))
        print(f"[🔑 GEMINI] Key 1 hazır → model: {GEMINI_MODEL_NAME}")

    if GEMINI_API_KEY_2:
        llm2 = ChatGoogleGenerativeAI(
            google_api_key=GEMINI_API_KEY_2,
            model=GEMINI_MODEL_NAME,
            temperature=temperature,
            max_retries=0,
            callbacks=[_GeminiKeyLogger("Gemini Key 2")],
        )
        available.append(("Gemini Key 2", llm2))
        print(f"[🔑 GEMINI] Key 2 hazır → model: {GEMINI_MODEL_NAME}")

    if len(available) == 1:
        name, llm = available[0]
        print(f"\n[🔑 GEMINI] Tek anahtarlı mod — {name}")
        logger.info("[GEMINI] Tek anahtarlı mod: %s", name)
        return llm

    primary_name, primary_llm = available[0]
    fallbacks = [llm for _, llm in available[1:]]
    chain = primary_llm.with_fallbacks(
        fallbacks=fallbacks,
        exceptions_to_handle=(Exception,),
    )

    names_str = " → ".join(n for n, _ in available)
    print(f"\n[🔑 GEMINI] Çift anahtarlı şelale kuruldu: {names_str}")
    logger.info("[GEMINI] Şelale zinciri: %s", names_str)
    return chain


def create_llm(
    provider: Optional[str] = None,
    temperature: float = 0.2,
) -> BaseChatModel:
    """Geriye uyumluluk: artık her zaman fallback zinciri döner."""
def get_active_provider() -> tuple[str, str]:
    """
    Sidebar için (provider_key, model_label) çifti döner.
    Çift anahtarlı modda "gemini_dual", tek anahtarlı modda "gemini_single" döner.
    """
    if not GEMINI_API_KEY_1 and not GEMINI_API_KEY_2:
        raise EnvironmentError(
            "Gemini API anahtarı bulunamadı!\n"
            ".env dosyasına GEMINI_API_KEY_1 ve/veya GEMINI_API_KEY_2 ekleyin."
        )

    key_count = sum(1 for k in (GEMINI_API_KEY_1, GEMINI_API_KEY_2) if k)
    mode = "gemini_dual" if key_count == 2 else "gemini_single"
    label = GEMINI_MODEL_NAME
    return mode, label
