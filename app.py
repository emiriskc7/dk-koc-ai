"""
app.py — Streamlit Premium Kullanıcı Arayüzü
=============================================
Çalıştırmak için:  streamlit run app.py

Özellikler:
  • WhatsApp/Trendyol tarzı şık sohbet balonları (Custom CSS)
  • Kelime kelime yazma animasyonu (st.write_stream + LCEL streaming)
  • API hataları için kibar kullanıcı mesajları (kırmızı Traceback yok)
  • Sidebar: model bilgisi, bilgi tabanı istatistikleri, sohbet temizleme
  • Hoş geldin ekranı: örnek sorular ile kullanıcıyı yönlendirir
"""

# ─── Diğer TÜM importlardan önce .env'i zorla yükle ──────────
import os
from pathlib import Path
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent
_dot_env       = _project_root / ".env"
_dot_env_ex    = _project_root / ".env.example"

if _dot_env.exists():
    load_dotenv(dotenv_path=_dot_env, override=True)
elif _dot_env_ex.exists():
    load_dotenv(dotenv_path=_dot_env_ex, override=True)
# ─────────────────────────────────────────────────────────────

import sys
import uuid
import random
import re
import logging
import traceback

import streamlit as st

# Proje kökünü path'e ekle
sys.path.insert(0, str(_project_root))

from config import APP_TITLE, APP_DESCRIPTION, MAX_CHAT_HISTORY
from src.chain import clear_session_history, stream_answer, STREAM_RETRY_SENTINEL, load_pdf_context, generate_quick_questions_with_pdf
from src.llm_factory import get_active_provider
from src.retriever import get_vectorstore_stats
from src.quick_questions import FLAT_LIST as _ALL_QUICK_QUESTIONS, SAMPLE_QUESTIONS as _SAMPLE_QUESTIONS

logging.basicConfig(level=logging.INFO)

# ═════════════════════════════════════════════════════════════
# 1. SAYFA AYARLARI  (st.set_page_config en üstte olmalı)
# ═════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Koçluk Asistanı",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "YKS koçluk bilgi tabanı üzerine kurulu RAG chatbot.",
    },
)

# ═════════════════════════════════════════════════════════════
# 2. CUSTOM CSS — Premium Chat Bubble Tasarımı
# ═════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* ── GENEL ARKA PLAN ─────────────────────────────────── */
    .stApp {
        background: linear-gradient(145deg, #f0f4ff 0%, #fafafa 60%, #f0fff8 100%);
    }

    /* ── BAŞLIK DEGRADESİ ────────────────────────────────── */
    h1 {
        background: linear-gradient(90deg, #6366f1, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    /* ── SOHBET MESAJ KARTI (her iki taraf) ──────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 4px 8px;
        margin: 6px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.05);
        animation: fadeSlideUp 0.3s ease;
        transition: box-shadow 0.2s ease;
    }
    [data-testid="stChatMessage"]:hover {
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.10);
    }

    /* ── KULLANICI MESAJI — mor-mavi degrade sol border ──── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        border-left: 4px solid #6366f1;
    }

    /* ── ASİSTAN MESAJI — yeşil aksan, beyaz kart ────────── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #ffffff;
        border-left: 4px solid #10b981;
    }

    /* ── GİRİŞ ALANI ──────────────────────────────────────── */
    [data-testid="stChatInput"] {
        border-radius: 28px !important;
        border: 2px solid #6366f1 !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.18) !important;
        font-size: 15px !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #10b981 !important;
        box-shadow: 0 4px 24px rgba(16, 185, 129, 0.22) !important;
    }

    /* ── HOŞ GELDİN KARTI ─────────────────────────────────── */
    .welcome-card {
        background: linear-gradient(135deg, #6366f1 0%, #10b981 100%);
        border-radius: 20px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.28);
    }
    .welcome-card h3 {
        margin: 0 0 8px 0;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .welcome-card p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* ── ÖRNEK SORU BUTONLARI ─────────────────────────────── */
    .sample-question {
        display: inline-block;
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px 4px 4px 0;
        font-size: 0.85rem;
        color: #4f46e5;
        cursor: pointer;
        transition: all 0.2s;
    }
    .sample-question:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: #6366f1;
    }

    /* ── SİDEBAR ──────────────────────────────────────────── */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 50%, #1e3a5f 100%);
        padding: 1.5rem 1.2rem;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e8eaf6 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e8eaf6;
    }

    /* ── SİDEBAR BİLGİ KARTI ──────────────────────────────── */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 10px 0;
        backdrop-filter: blur(8px);
    }
    .sidebar-card .card-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #a5b4fc !important;
        margin-bottom: 4px;
    }
    .sidebar-card .card-value {
        font-size: 0.92rem;
        color: #f1f5f9 !important;
        font-weight: 500;
    }
    .sidebar-card .card-sub {
        font-size: 0.78rem;
        color: #94a3b8 !important;
        margin-top: 2px;
    }

    /* ── UYARI KUTUSU (vector store bulunamadı) ───────────── */
    .setup-warning {
        background: linear-gradient(135deg, #fff7ed, #fef3c7);
        border: 2px solid #f59e0b;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 20px 0;
    }

    /* ── GEÇİŞ ANİMASYONU ─────────────────────────────────── */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 3. SESSION STATE BAŞLATMA
# ═════════════════════════════════════════════════════════════

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Örnek soru butonlarının tetiklediği bekleyen soruyu taşır
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# LLM'in ürettigi dinamik devam soruları (son cevaptan parse edilir)
if "ai_suggestions" not in st.session_state:
    st.session_state.ai_suggestions = []

# Sidebar hızlı sorular için rastgele tohum
if "quick_q_seed" not in st.session_state:
    st.session_state.quick_q_seed = random.randint(0, 9999)

if "provider" not in st.session_state:
    try:
        provider, model = get_active_provider()
        st.session_state.provider = provider
        st.session_state.model = model
    except EnvironmentError as exc:
        print(f"\n[APP] API anahtarı/ortam hatası: {exc}")
        traceback.print_exc()
        st.error(str(exc))
        st.stop()

# ───────────────────────────────────────────────────────────────
# SHARED UI HELPERS  (– sidebar ve welcome card her ikisi de kullanır)
# ───────────────────────────────────────────────────────────────

def _set_pending_question(question: str) -> None:
    """on_click callback — seçilen soruyu session_state'e yazar."""
    st.session_state.pending_question = question


# ── SIFIR ÇÖKME (ZERO-CRASH) ÖNERİ PARSE YARDIMCILARI ──────────
# LLM etiket üretmezse, yarım bırakırsa, köşeli parantezi
# unutursa hiçbir şey çökmez; ana cevap her zaman görünür.

def _safe_parse_suggestions(raw: str) -> list[str]:
    """
    raw metinden [ÖNERİ: ...] etiketlerini güvenle çıkarır.
    Herhangi bir hata → boş liste döner, uygulama devam eder.
    """
    if not isinstance(raw, str) or not raw:
        return []
    try:
        matches = re.findall(r'\[ÖNERİ:\s*(.*?)\]', raw, re.DOTALL)
        return [m.strip() for m in matches if m.strip()][:3]
    except Exception as exc:
        print(f"[PARSE] _safe_parse_suggestions hatası: {exc}")
        traceback.print_exc()
        return []


def _safe_clean_text(raw: str) -> str:
    """
    raw metinden [ÖNERİ: ...] etiketlerini silerek temiz metni döner.
    Herhangi bir hata → raw metnin kendisini döner, cevap kaybolmaz.
    """
    if not isinstance(raw, str):
        return ""
    try:
        return re.sub(r'\s*\[ÖNERİ:.*?\]\s*', '', raw, flags=re.DOTALL).strip()
    except Exception as exc:
        print(f"[PARSE] _safe_clean_text hatası: {exc}")
        traceback.print_exc()
        return raw.strip()


# _ALL_QUICK_QUESTIONS ve _SAMPLE_QUESTIONS artık src.quick_questions'dan import ediliyor

# ═════════════════════════════════════════════════════════════
# 4. SIDEBAR
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")
    st.divider()

    # --- Model Bilgisi ---
    _p = st.session_state.provider
    if _p == "gemini":
        provider_label = "🔵 Google Gemini (Şelale)"
    else:
        provider_label = "🔵 Google Gemini"
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="card-label">🤖 Aktif Model</div>
            <div class="card-value">{provider_label}</div>
            <div class="card-sub">{st.session_state.model}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Bilgi Tabanı İstatistikleri ---
    stats = get_vectorstore_stats()
    if stats["status"] == "ok":
        status_icon = "🟢"
        status_text = "Hazır"
    elif stats["status"] == "not_found":
        status_icon = "🔴"
        status_text = "Bulunamadı"
    else:
        status_icon = "🟡"
        status_text = "Hata"

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="card-label">📚 Bilgi Tabanı</div>
            <div class="card-value">{status_icon} {status_text}</div>
            <div class="card-sub">
                {stats['doc_count']:,} vektör &nbsp;|&nbsp;
                Eşik: {stats['score_threshold']}<br>
                Embedding: {stats['embedding_model']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Sohbeti Temizle ---
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        clear_session_history(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())  # yeni oturum başlat
        st.rerun()

    # --- Sohbet İstatistiği ---
    msg_count = len(st.session_state.messages)
    if msg_count > 0:
        st.caption(f"Bu oturumda {msg_count // 2} soru soruldu.")

    st.divider()

    # ─── HIZLI SORULAR (PDF'lerden üretime dayalı + statik fallback) ───────
    st.markdown("### 💡 Hızlı Sorular")
    
    # ─ PDF verilerini yükle ve PDF-tabanlı sorular üret
    with st.spinner("📚 Sorular hazırlanıyor..."):
        pdf_ctx = load_pdf_context()
        pdf_questions = generate_quick_questions_with_pdf(pdf_ctx) if pdf_ctx else []
    
    # ─ Gösterilecek sorular: PDF'den varsa onları göster, yoksa statik havuzdan al
    if pdf_questions:
        shown_qs = pdf_questions
        st.caption("📖 Rehberlik verilerine dayanan sorular")
    else:
        rng = random.Random(st.session_state.quick_q_seed)
        shown_qs = rng.sample(_ALL_QUICK_QUESTIONS, 5)
        st.caption("💡 Önerilen koçluk soruları")
    
    for sq in shown_qs:
        st.button(
            sq,
            key=f"sidebar_q_{sq[:20]}",
            use_container_width=True,
            on_click=_set_pending_question,
            args=(sq,),
        )
    
    if st.button("🔄 Farklı Sorular", use_container_width=True, type="secondary"):
        st.session_state.quick_q_seed = random.randint(0, 9999)
        st.rerun()

# ═════════════════════════════════════════════════════════════
# 5. VEKTÖR MAĞAZASI UYARISI  (ingest.py çalıştırılmamışsa)
# ═════════════════════════════════════════════════════════════

if stats["status"] == "not_found":
    st.markdown(
        """
        <div class="setup-warning">
            <h3>⚠️ Kurulum Gerekiyor</h3>
            <p>Bilgi tabanı henüz oluşturulmamış.<br>
            Chatbot'u başlatmadan önce terminalden aşağıdaki komutu çalıştırın:</p>
            <pre style="background:#1e293b;color:#a5b4fc;padding:10px 14px;
                        border-radius:8px;margin-top:12px;">python ingest.py</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ═════════════════════════════════════════════════════════════
# 6. ANA BAŞLIK
# ═════════════════════════════════════════════════════════════

st.title(APP_TITLE)
st.caption(APP_DESCRIPTION)

# ═════════════════════════════════════════════════════════════
# 7. HOŞ GELDİN KARTI  (mesaj yokken gösterilir)
# ═════════════════════════════════════════════════════════════

# Hoş geldin kartını: mesaj YOK ve bekleyen soru da YOK iken göster
if not st.session_state.messages and not st.session_state.pending_question:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>Merhaba! 👋</h3>
            <p>
                Koçluk bilgi tabanına dayalı YKS hazırlık asistanınım.<br>
                Deneyimli koçların gerçek tavsiyelerini seninle paylaşmak için buradayım.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**💡 Örnek sorular:**")
    cols = st.columns(2)
    for i, q in enumerate(_SAMPLE_QUESTIONS):
        with cols[i % 2]:
            # on_click → callback önce koşar, pending_question set edilir,
            # sayfa yeniden render edilir, aşağıdaki active_prompt bloğu işler.
            st.button(
                q,
                key=f"sample_{i}",
                use_container_width=True,
                on_click=_set_pending_question,
                args=(q,),
            )

    st.divider()

# ═════════════════════════════════════════════════════════════
# 8. SOHBET GEÇMİŞİNİ GÖSTER
# ═════════════════════════════════════════════════════════════

# Bellek yönetimi: aşırı büyümeyi önle
if len(st.session_state.messages) > MAX_CHAT_HISTORY * 2:
    st.session_state.messages = st.session_state.messages[-(MAX_CHAT_HISTORY * 2):]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ═════════════════════════════════════════════════════════════
# 8b. DİNAMİK DEVAM SORULARI  (LLM'den parse edilen ÖNERİ etiketleri)
# ═════════════════════════════════════════════════════════════

_DEFAULT_SUGGESTIONS = [
    "Öğrencim geometride sıfırdan başlıyor, ilk 2 haftanın planını nasıl çıkarırım?",
    "Öğrencim denemede 20 net yapıyor ama hiç gelişmiyor, kök nedeni nasıl buluruz?",
    "Öğrencim burnout yaşıyor, bu kriz anında nasıl bir koçluk seansı yapmalıyım?",
]

# Hangi öneriler gösterilecek: LLM ürettiyse onu, yoksa varsayılan
_active_suggestions: list[str] = (
    st.session_state.ai_suggestions
    if st.session_state.ai_suggestions
    else (_DEFAULT_SUGGESTIONS if not st.session_state.messages else [])
)

if _active_suggestions:
    st.markdown(
        "<div style='margin-top:8px;margin-bottom:4px;font-size:0.82rem;"
        "color:#6366f1;font-weight:600;'>&#128172; Devam sorusu önerileri</div>",
        unsafe_allow_html=True,
    )
    for _si, _sq in enumerate(_active_suggestions):
        st.button(
            f"💭 {_sq}",
            key=f"dyn_{len(st.session_state.messages)}_{_si}",
            use_container_width=True,
            on_click=_set_pending_question,
            args=(_sq,),
        )

# ═════════════════════════════════════════════════════════════
# 9. KULLANICI GİRDİSİ VE STREAMING YANIT
# ═════════════════════════════════════════════════════════════

# Aktif prompt kaynağı:
#   • Buton on_click → pending_question  (bu render'da tüketilir)
#   • Kılavye    → st.chat_input       (klasik akış)
active_prompt: str | None = st.session_state.pop("pending_question", None)

if typed := st.chat_input("Hocam, öğrencileriniz için ne sormak istersiniz?"):
    active_prompt = typed  # klavye girdisi her zaman önceliklidir

if active_prompt:

    # Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    # ─ Streaming: token'ları topla, canlı göster, ÖNERİ etiketlerini gizle ─
    with st.chat_message("assistant"):
        _ph = st.empty()       # dinamik placeholder
        _raw = ""              # ham çıktı (ÖNERİ etiketleri dahil)
        _suggestions: list[str] = []
        _clean = ""

        try:
            for _chunk in stream_answer(
                query=active_prompt,
                session_id=st.session_state.session_id,
                provider=st.session_state.provider,
            ):
                # ── Rate-limit retry sinyali: birikmiş _raw'ı sıfırla ──
                if _chunk == STREAM_RETRY_SENTINEL:
                    _raw = ""
                    _ph.markdown("⏳ *Yeniden bağlanıyor, lütfen bekleyin...*")
                    continue

                _raw += _chunk

                # ÖNERİ etiketleri cevabın sonuna gelir; canlı gösterimde gizle
                try:
                    _live = re.sub(r'\[ÖNERİ:.*', '', _raw, flags=re.DOTALL).rstrip()
                except Exception as _re_exc:
                    print(f"[APP] Canlı regex hatası: {_re_exc}")
                    _live = _raw
                _ph.markdown(_live + " ●")

            # ── Stream bitti: ÖNERİ etiketlerini SIFIR ÇÖKME ile parse et ──
            _suggestions = _safe_parse_suggestions(_raw)
            _clean       = _safe_clean_text(_raw)
            _ph.markdown(_clean)

        except Exception as _stream_exc:
            print("\n" + "=" * 60)
            print(f"[APP] 🔥 STREAMING HATASI: {type(_stream_exc).__name__}: {_stream_exc}")
            traceback.print_exc()
            print("=" * 60 + "\n")
            _suggestions = []
            _clean = _raw.strip()
            if _clean:
                _ph.markdown(_clean)
            else:
                _ph.empty()

            st.error("🚨 Gemini motorlarında (Key 1 ve Key 2) bir sorun oluştu.")
            st.error(f"🔥 Asıl Hata Sebebi: {str(_stream_exc)}")
            st.code(traceback.format_exc())
            # KRİTİK: Bu satır olmazsa hemen alttaki st.rerun() hatayı siler!
            st.stop()

    # Temiz yanıtı geçmişe kaydet; ÖNERİ etiketleri kaydedilmez
    st.session_state.messages.append({"role": "assistant", "content": _clean})
    st.session_state.ai_suggestions = [s.strip() for s in _suggestions if s.strip()][:3]
    st.rerun()  # devam sorularını hemen render et
