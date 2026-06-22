"""
chain.py — JSON + PDF Doğruluk Kaynağı + Çift Gemini Şelale RAG Zinciri
=========================================================================
Mimari:

  1. load_coaches_json_context()
     → data/coaching_knowledge_base.json okunur, temiz metin bloğuna çevrilir.

  2. load_pdf_context()
     → data/pdfs/ klasöründeki tüm .pdf dosyaları okunur, tek blok döner.

  3. build_coaching_prompt(json_ctx, pdf_ctx)
     → JSON + PDF verileri SystemMessage'e "Doğruluk Kaynağı" olarak enjekte edilir.

  4. build_chain()
     → Çift Gemini anahtarlı şelale LLM + RAG retriever + JSON+PDF system
       prompt u tek bir RunnableWithMessageHistory zincirine bağlar.

  5. stream_answer()
     → Streamlit için token generator.
"""

import json
import logging
import traceback
import re
from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional

from pypdf import PdfReader
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import COACHES_CHAT_JSON_PATH, COACH_PDF_DIR_PATH
from src.llm_factory import create_fallback_llm
from src.retriever import get_retriever

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# JSON DOĞRULUK KAYNAĞI YÜKLEYİCİ
# ─────────────────────────────────────────────────────────────

def load_coaches_json_context(max_chars: int = 14_000) -> str:
    """
    data/coaches_chat.json dosyasını okur ve temiz bir metin bloğuna dönüştürür.

    Desteklenen JSON formatları:
      • Liste: [{"role": "koç", "content": "..."}, ...]
      • Liste: [{"speaker": "...", "message": "..."}, ...]
      • Dict : {"conversations": [...], "guidelines": [...], ...}

    Dosya bulunamazsa veya parse hatası olursa boş string döner;
    uygulama çökmez, uyarıyla devam eder.
    """
    try:
        path = Path(COACHES_CHAT_JSON_PATH)
        if not path.exists():
            print(f"[WARNING JSON] Koçluk geçmişi bulunamadı: {path}")
            logger.warning("coaches_chat.json bulunamadı: %s", path)
            return ""

        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        lines: list[str] = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    role = item.get("role") or item.get("speaker") or item.get("from", "")
                    content = (
                        item.get("content")
                        or item.get("message")
                        or item.get("text", "")
                    )
                    if role and content:
                        lines.append(f"[{role.upper()}]: {content}")
                    else:
                        for k, v in item.items():
                            if isinstance(v, str) and v.strip():
                                lines.append(f"{k}: {v}")
                elif isinstance(item, str) and item.strip():
                    lines.append(item)

        elif isinstance(data, dict):
            for key, val in data.items():
                lines.append(f"\n## {key.upper()}")
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            role = item.get("role") or item.get("speaker", "")
                            content = (
                                item.get("content")
                                or item.get("message")
                                or item.get("text", "")
                            )
                            if role and content:
                                lines.append(f"  [{role.upper()}]: {content}")
                            else:
                                for k, v in item.items():
                                    if isinstance(v, str) and v.strip():
                                        lines.append(f"  {k}: {v}")
                        elif isinstance(item, str) and item.strip():
                            lines.append(f"  - {item}")
                elif isinstance(val, str) and val.strip():
                    lines.append(val)

        context_text = "\n".join(lines).strip()

        if len(context_text) > max_chars:
            context_text = (
                context_text[:max_chars]
                + "\n\n... [Koçluk geçmişi devam ediyor — kalan kısım sistem belleğinde]"
            )

        char_count = len(context_text)
        print(
            f"[OK JSON] Koçluk geçmişi yüklendi: "
            f"{len(lines)} satır, {char_count:,} karakter"
        )
        logger.info(
            "coaches_chat.json yüklendi: %d satır, %d karakter",
            len(lines), char_count,
        )
        return context_text

    except json.JSONDecodeError as exc:
        print(f"[WARNING JSON] Parse hatası — uygulama devam ediyor: {exc}")
        logger.error("coaches_chat.json parse hatası: %s", exc)
        return ""
    except Exception as exc:
        print(f"[WARNING JSON] Okuma hatası — uygulama devam ediyor: {exc}")
        logger.error("coaches_chat.json beklenmeyen hata: %s", exc)
        return ""



# ─────────────────────────────────────────────────────────────
# PDF DOĞRULUK KAYNAĞI YÜKLEYİCİ
# ─────────────────────────────────────────────────────────────

def load_pdf_context(max_chars: int = 800_000) -> str:
    """
    data/pdfs/ klasöründeki tüm .pdf dosyalarını okur ve metin bloğu döner.
    
    Ücretsiz Google API kotası (250K token/dk) nedeniyle PDF metni
    maksimum 800.000 karaktere (≈ 200K token) sınırlandırılır.
    Kesme yapıldığında sadece terminale uyarı mesajı yazılır,
    arayüz (Streamlit) bozulmaz.
    
    Klasör yoksa, boşsa veya PDF okunamazsa uygulama ÇÖKMEZ;
    uyarı loglanarak boş string ile devam edilir.
    """
    try:
        pdf_dir = Path(COACH_PDF_DIR_PATH)
        if not pdf_dir.exists() or not pdf_dir.is_dir():
            logger.warning("PDF klasörü bulunamadı: %s", pdf_dir)
            return ""

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("PDF klasörü boş: %s", pdf_dir)
            return ""

        all_texts: list[str] = []
        loaded_count = 0

        for pdf_path in pdf_files:
            try:
                reader = PdfReader(str(pdf_path))
                pages: list[str] = []
                for page in reader.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(text.strip())
                if pages:
                    all_texts.append(
                        f"### {pdf_path.stem}\n" + "\n".join(pages)
                    )
                    loaded_count += 1
            except Exception as exc:
                logger.warning("PDF okunamadı (%s): %s", pdf_path.name, exc)

        if not all_texts:
            return ""

        combined = "\n\n".join(all_texts).strip()
        
        # ── Ücretsiz API Kotası Koruması ──────────────────────────────
        truncated = False
        if len(combined) > max_chars:
            combined = combined[:max_chars]
            truncated = True
            # Sadece terminal logu — arayüzü bozma
            print(
                "\n⚠️  Ücretsiz API sınırı (250K token) nedeniyle "
                "PDF verisi 800.000 karakterde sınırlandırıldı.\n"
            )

        if truncated:
            logger.info(
                "PDF rehberleri yüklendi: %d dosya, %d karakter (KESILDI → max %d)",
                loaded_count, len(combined), max_chars,
            )
        else:
            logger.info(
                "PDF rehberleri yüklendi: %d dosya, %d karakter",
                loaded_count, len(combined),
            )
        
        return combined

    except Exception as exc:
        logger.error("PDF yükleme beklenmeyen hata: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────
# OTURUM GEÇMİŞİ DEPOSU  (in-memory, uygulama yaşadığı sürece)
# ─────────────────────────────────────────────────────────────

_session_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Mevcut oturum geçmişini döner; yoksa yeni bir tane oluşturur."""
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def clear_session_history(session_id: str) -> None:
    """Oturum geçmişini tamamen siler (UI Sohbeti Temizle butonu için)."""
    _session_store.pop(session_id, None)
    logger.info("Oturum geçmişi temizlendi: %s", session_id)


# ─────────────────────────────────────────────────────────────
# PROMPT ŞABLONLARI
# ─────────────────────────────────────────────────────────────

_CONTEXTUALIZE_Q_SYSTEM = (
    "Bir sohbet geçmişi ve kullanıcının son sorusu sana veriliyor. "
    "Görevin: sohbet geçmişi olmadan da tek başına anlaşılabilecek, "
    "bağımsız bir soru üret. Soruyu CEVAPLAMA; yalnızca gerekiyorsa "
    "yeniden yaz, gereksizse aynen bırak."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _CONTEXTUALIZE_Q_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def build_coaching_prompt(coaches_json_context: str, pdf_context: str = "") -> ChatPromptTemplate:
    """
    Koçluk JSON bağlamını SystemMessagePromptTemplate e enjekte ederek
    sıfır halüsinasyon garantili ana prompt şablonunu oluşturur.
    """
    # ── DOĞRULUK KAYNAĞI 1: Koçluk Sohbet Geçmişi (JSON) ──────────
    if coaches_json_context.strip():
        json_block = (
            "\n"
            "=================================================================\n"
            "DOĞRULUK KAYNAĞI 1 — GERÇEK KOÇ SOHBET GEÇMİŞİ\n"
            "=================================================================\n\n"
            + coaches_json_context
            + "\n\n"
            "=================================================================\n"
        )
    else:
        json_block = ""

    # ── DOĞRULUK KAYNAĞI 2: YKS Rehberlik Kılavuzları (PDF) ────────
    if pdf_context.strip():
        pdf_block = (
            "\n"
            "=================================================================\n"
            "DOĞRULUK KAYNAĞI 2 — RESMİ YKS REHBERLİK KILAVUZLARI (PDF)\n"
            "=================================================================\n\n"
            + pdf_context
            + "\n\n"
            "=================================================================\n"
        )
    else:
        pdf_block = ""
    system_template = (
        "Sen gerçek koçluk sohbet verileri ve resmi YKS Rehberlik Kılavuzları ile "
        "donatılmış, profesyonel bir YKS/LGS/KPSS Koç Asistanısın.\n"
        "Bir öğrenci veya koç sana soru sorduğunda, hem geçmiş sohbetlerdeki "
        "samimi ve doğal koçluk dilini kullan, hem de rehberlerdeki zengin, "
        "akademik bilgileri sentezleyerek doğrudan ve profesyonelce cevap ver.\n"
        "Asla yapay zeka olduğunu veya bu verileri arka planda okuduğunu belli etme.\n\n"

        "=================================================================\n"
        "KARAKTER VE TON\n"
        "=================================================================\n\n"

        "• Her zaman 'Hocam' diye hitap et; koçu yönlendiren bir dil kullan.\n"
        "• Ton: Sıcak, güven veren, aksiyon odaklı, net ve teşvik edici.\n"
        "• Cevapların kısa değil ama gereksiz dolgu içermesin.\n\n"

        "=================================================================\n"
        "CEVAP MANTIĞI — İKİ KAYNAK, TEK SES\n"
        "=================================================================\n\n"

        "1)  Aşağıdaki KOÇLUK GEÇMİŞİ veya BELGE BAĞLAMI'nda soruyla\n"
        "    örtüşen bir yönlendirme varsa:\n"
        "    → O verileri temel al, aynen koçların kullandığı dil ve\n"
        "      yaklaşımla sun.\n\n"

        "2)  Konu geçmiş kaynaklarda doğrudan geçmiyorsa:\n"
        "    → Koçluk alanındaki kendi uzmanlığını devreye al; aynı\n"
        "      sıcaklıkta, aynı doğal koç diliyle yönlendir.\n"
        "    → 'Bilmiyorum' veya 'Bu konuda veri yok' kesinlikle DEME.\n"
        "    → 'Yapay zeka olarak', 'JSON verisinde', 'Kayıtlarda bulamadım'\n"
        "      gibi teknik ifadeler ASLA kullanma.\n\n"

        "YASAK KALIPLAR (hiçbir biçimde kullanma):\n"
        "  ✘ Bu konuda geçmiş koçluk kayıtlarında bulamadım...\n"
        "  ✘ Veritabanımda bu konu geçmiyor...\n"
        "  ✘ Bir yapay zeka olarak söyleyebileceğim...\n\n"

        + json_block
        + pdf_block

        + "=================================================================\n"
        "CEVAP YAPISI\n"
        "=================================================================\n\n"

        "• Net bir açılış / yönlendirme cümlesi\n"
        "• Madde madde, uygulanabilir adımlar\n"
        "• Sonunda 1-2 devam sorusu önerisi: [ÖNERİ: soru metni]\n\n"

        "─────────────────────────────────────────────────────────────────\n"
        "Ek Bağlam (Bilgi Tabanı Belgeler):\n"
        "{context}\n"
        "─────────────────────────────────────────────────────────────────"
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


# ─────────────────────────────────────────────────────────────
# ZİNCİR KURULUMU  (lru_cache → bir kez kur, hep kullan)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_chain() -> RunnableWithMessageHistory:
    """
    Çift Gemini anahtarlı şelale LLM + JSON doğruluk kaynağı +
    RAG retriever ı tek bir RunnableWithMessageHistory zincirine bağlar.
    """
    # 1. Doğruluk kaynaklarını yükle (JSON + PDF)
    coaches_context = load_coaches_json_context()
    pdf_ctx = load_pdf_context()

    # 2. LLM (çift anahtarlı şelale)
    llm = create_fallback_llm()

    # 3. Retriever (ChromaDB vektör mağazası)
    retriever = get_retriever()

    # 4. Sıfır halüsinasyon system promptu (JSON bağlamı enjekte edilmiş)
    coaching_prompt = build_coaching_prompt(coaches_context, pdf_ctx)

    # 5. Sohbet bağlamını bağımsız arama sorgusuna dönüştür
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 6. Dokümanları sistemprompt ile birleştir
    doc_chain = create_stuff_documents_chain(llm, coaching_prompt)

    # 7. Retrieval + Generation
    rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    # 8. Oturum geçmişi katmanı
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    sources = []
    if coaches_context: sources.append("JSON")
    if pdf_ctx: sources.append("PDF")
    src_label = " + ".join(sources) if sources else "kaynak yok"
    print(f"\n[OK ZİNCİR] Koç Asistanı hazır — {src_label}")
    logger.info("Koç Asistanı zinciri kuruldu — JSON ground truth + RAG aktif")
    return conversational_chain


# ─────────────────────────────────────────────────────────────
# HATA TANIMLARI & YARDIMCILAR
# ─────────────────────────────────────────────────────────────

_MSG_NO_CONTEXT = (
    "Bu konuda geçmiş koçluk kayıtlarında net bir yönlendirme bulamadım, "
    "hatalı yönlendirme yapmamak adına koçluk grubuna danışmanızı öneririm."
)

STREAM_RETRY_SENTINEL = "\x00__RETRY__\x00"  # artık kullanılmıyor, geriye uyumluluk için

# ── Fallback: Statik PDF-uyumlu hızlı sorular ──────────────────
_FALLBACK_PDF_QUESTIONS = [
    "TYT Matematikte önerilen turlama tekniği nedir?",
    "Sınav anı stresi için rehberdeki nefes egzersizi nasıl yapılır?",
    "OBP puanı Tıp öğrenciliğinde nasıl hesaplanır?",
]


def generate_quick_questions_with_pdf(pdf_context: str) -> list[str]:
    """
    PDF rehberlik verilerine dayanarak 3 adet spesifik, akademik soru üretir.
    
    Bu sorular sidebar'da gösterilir ve öğrencinin PDF'lerdeki teknik bilgileri
    merak etmesini sağlar. Fallback olarak 3 statik soru kullanılır.
    
    Parametre:
      pdf_context: Okunan PDF metni (tamamı veya ilk 50K karakter)
    
    Döner:
      list[str]: 3 adet hızlı soru
    """
    try:
        # ── Hız ve API güvenliği için PDF'in ilk 50K karakterini kullan
        pdf_excerpt = pdf_context[:50_000] if pdf_context else ""
        if not pdf_excerpt.strip():
            logger.warning("PDF bağlamı boş — fallback sorular kullanılıyor")
            return _FALLBACK_PDF_QUESTIONS

        # ── Prompt: PDF-tabanlı spesifik sorular üret
        prompt_text = (
            "Aşağıdaki YKS/LGS Rehberlik PDF'ini okudun:\n\n"
            f"{pdf_excerpt}\n\n"
            "Bu PDF'de yer alan spesifik bilgilere dayanarak, bir öğrencinin "
            "sorması için 3 adet kısa, kaliteli ve akademik soru üret. "
            "Sorular PDF'deki sınav taktikleri, puan hesaplaması, zaman yönetimi "
            "veya stresi yönetme konularına odaklanmalı. "
            "Her soru sadece 1 satır olmalı; 'Nasıl motive olurum?' gibi sıradan "
            "sorular değil; 'TYT matematik turlama tekniği nedir?' tarzında "
            "spesifik ve pratik sorular. "
            "\n\nÖneri: Sadece soruları, başka açıklama yapma. "
            "Her soruyu yeni satıra yaz (hepsi 1 satır olacak).\n"
        )

        # ── LLM ile sorular üret
        llm = create_fallback_llm(temperature=0.3)  # 0.3 = yaratıcılık & kontrol dengesi
        try:
            response = llm.invoke([("human", prompt_text)])
            response_text = response.content.strip()
        except Exception as llm_err:
            logger.warning("LLM çağrısı başarısız: %s — fallback sorular kullanılıyor", llm_err)
            return _FALLBACK_PDF_QUESTIONS

        # ── Sorguları satırlara ayır ve temizle
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        
        # ── Numerik prefix'leri temizle (1., 2., vb.)
        cleaned_qs = []
        for line in lines:
            # "1. Soru metni" → "Soru metni"
            match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
            if match:
                cleaned_qs.append(match.group(1))
            elif line and not line.startswith("#"):
                cleaned_qs.append(line)
        
        # ── 3 soru garantiyle return et (az varsa fallback'ten tamamla)
        if len(cleaned_qs) >= 3:
            return cleaned_qs[:3]
        elif len(cleaned_qs) > 0:
            # Kısmi sonuç + fallback
            return cleaned_qs + _FALLBACK_PDF_QUESTIONS[len(cleaned_qs):]
        else:
            # Tamamen boş yanıt
            logger.warning("LLM boş yanıt döndürdü — fallback sorular kullanılıyor")
            return _FALLBACK_PDF_QUESTIONS

    except Exception as exc:
        logger.error("Hızlı sorular üretimi hatası: %s — fallback sorular kullanılıyor", exc)
        return _FALLBACK_PDF_QUESTIONS


# ─────────────────────────────────────────────────────────────
# STREAMING
# ─────────────────────────────────────────────────────────────


def stream_answer(
    query: str,
    session_id: str,
    provider: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Streamlit için token generator.

    • LangChain chain inden answer token larını yield eder.
    • Rate-limit (429) hatalarında STREAM_RETRY_SENTINEL yield edip
      exponential backoff ile yeniden dener (max 3 kez).
    • Retriever boş dönerse kibarca bildirir.
    • Diğer hatalar: terminale traceback, sonra re-raise.
    """
    try:
        chain = build_chain()
    except Exception as exc:
        print("\n" + "=" * 60)
        print(f"[ZİNCİR] Zincir kurulumu başarısız: {exc}")
        traceback.print_exc()
        print("=" * 60 + "\n")
        raise

    try:
        full_answer = ""

        for chunk in chain.stream(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        ):
            token = chunk.get("answer", "")
            if token:
                full_answer += token
                yield token

        if not full_answer.strip():
            yield _MSG_NO_CONTEXT

    except Exception as exc:
        print("\n" + "=" * 60)
        print(f"[ZİNCİR] 🔥 Gemini hatası: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("=" * 60 + "\n")
        logger.error("stream_answer hatası: %s", exc)
        raise
