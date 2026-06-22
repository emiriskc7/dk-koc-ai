"""
api.py — YKS/LGS Koçluk Asistanı · FastAPI Backend
=====================================================
Streamlit arayüzüne dokunmadan mevcut RAG zincirini REST API olarak sunar.

Çalıştırmak için:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Swagger (OpenAPI) dokümantasyonu:
    http://localhost:8000/docs

Redoc dokümantasyonu:
    http://localhost:8000/redoc

Endpointler:
  POST /chat           → Tam yanıt (JSON)
  POST /chat/stream    → Kelime kelime Server-Sent Events (SSE) akışı
  GET  /health         → Servis canlılık kontrolü
  DELETE /session/{id} → Oturum geçmişini temizle
"""


import os
from pathlib import Path
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent
_dot_env      = _project_root / ".env"
_dot_env_ex   = _project_root / ".env.example"

if _dot_env.exists():
    load_dotenv(dotenv_path=_dot_env, override=True)
elif _dot_env_ex.exists():
    load_dotenv(dotenv_path=_dot_env_ex, override=True)


import sys
import uuid
import logging
import asyncio
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


sys.path.insert(0, str(_project_root))

from src.chain import build_chain, stream_answer, clear_session_history


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="YKS/LGS Koçluk Asistanı API",
    description=(
        "Gerçek koçluk verileri (JSON + PDF) ve Gemini tabanlı RAG mimarisi "
        "üzerine kurulu YKS/LGS koçluk chatbot'unun REST API arayüzü.\n\n"
        "**Akış (Streaming):** `/chat/stream` endpointi Server-Sent Events (SSE) "
        "protokolü ile yanıtı kelime kelime iletir.\n\n"
        "**Oturum Yönetimi:** Her istemci `session_id` göndererek bağımsız "
        "sohbet geçmişi oluşturabilir."
    ),
    version="1.0.0",
    contact={
        "name": "Koçluk Asistanı Ekibi",
    },
    license_info={
        "name": "Özel Kullanım",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS 
# ─────────────────────────────────────────────────────────────────────────────
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELLER
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Sohbet isteği şeması."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Öğrenci veya koçun gönderdiği mesaj.",
        examples=["TYT matematikte nasıl çalışmalıyım?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Oturum kimliği. Boş bırakılırsa otomatik UUID üretilir ve "
            "yanıtta döner. Aynı oturumu sürdürmek için bir sonraki istekte "
            "bu değeri gönderin."
        ),
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


class ChatResponse(BaseModel):
    """Tam yanıt şeması (streaming dışı)."""

    answer: str = Field(..., description="Asistanın tam yanıtı.")
    session_id: str = Field(..., description="Aktif oturum kimliği.")


class SessionDeleteResponse(BaseModel):
    """Oturum silme onay şeması."""

    deleted: bool = Field(..., description="Silme işlemi başarılı mı?")
    session_id: str = Field(..., description="Silinen oturum kimliği.")


class HealthResponse(BaseModel):
    """Sağlık kontrolü şeması."""

    status: str = Field(..., description="Servis durumu: 'ok' veya 'error'.")
    model: str = Field(..., description="Aktif LLM modeli.")
    chain_ready: bool = Field(..., description="RAG zinciri hazır mı?")




def _resolve_session_id(session_id: Optional[str]) -> str:
    """
    Gelen session_id geçerliyse olduğu gibi döner;
    yoksa rastgele UUID üretir.
    """
    if session_id and session_id.strip():
        return session_id.strip()
    return str(uuid.uuid4())


async def _token_generator(query: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Senkron `stream_answer` generator'ını async SSE akışına dönüştürür.

    Her token `data: <token>\n\n` formatında gönderilir (SSE standardı).
    Akış sonunda `data: [DONE]\n\n` sinyali iletilir.
    """
    loop = asyncio.get_event_loop()

    gen = stream_answer(query=query, session_id=session_id)

    try:
        while True:
            # Senkron __next__ çağrısını executor üzerinde çalıştır
            token = await loop.run_in_executor(None, next, gen, None)
            if token is None:
                break
            # SSE formatı: data satırı + çift newline
            yield f"data: {token}\n\n"
    except StopIteration:
        pass
    except Exception as exc:
        logger.error("Streaming hatası: %s", exc)
        yield f"data: [ERROR] Yanıt üretilirken bir hata oluştu.\n\n"
    finally:
        yield "data: [DONE]\n\n"




@app.on_event("startup")
async def _startup_event():
    """
    Uygulama başladığında RAG zincirini ısındır.
    lru_cache sayesinde ikinci çağrıda maliyet sıfırdır.
    """
    logger.info("RAG zinciri ısındırılıyor...")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, build_chain)
        logger.info("RAG zinciri hazır.")
    except Exception as exc:
        logger.critical("Zincir başlatma hatası: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTLER
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Servis Sağlık Kontrolü",
    tags=["Sistem"],
)
async def health_check():
    """
    Servisin çalışıp çalışmadığını ve RAG zincirinin hazır olup olmadığını
    kontrol eder. Deployment sonrası canlılık (liveness) probe'u olarak kullanın.
    """
    from config import GEMINI_MODEL_NAME

    try:
        chain = await asyncio.get_event_loop().run_in_executor(None, build_chain)
        chain_ready = chain is not None
    except Exception:
        chain_ready = False

    return HealthResponse(
        status="ok" if chain_ready else "error",
        model=GEMINI_MODEL_NAME,
        chain_ready=chain_ready,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Sohbet Mesajı Gönder (Tam Yanıt)",
    tags=["Sohbet"],
)
async def chat(request: ChatRequest):
    """
    Öğrencinin mesajını alır ve **tüm yanıtı** tek seferde döner.

    - Yanıt asistan tarafından tamamen üretildikten sonra iletilir.
    - Uzun yanıtlar için `/chat/stream` endpointini tercih edin.
    - `session_id` belirtilmezse otomatik oluşturulur ve yanıtta döner.
    """
    session_id = _resolve_session_id(request.session_id)
    logger.info("Chat isteği — session: %s | mesaj: %.80s…", session_id, request.message)

    try:
        loop = asyncio.get_event_loop()

        # Tüm token'ları birleştir
        full_answer_parts: list[str] = []

        def _collect() -> str:
            for token in stream_answer(query=request.message, session_id=session_id):
                full_answer_parts.append(token)
            return "".join(full_answer_parts)

        answer = await loop.run_in_executor(None, _collect)

        if not answer.strip():
            raise HTTPException(status_code=500, detail="Asistan boş yanıt döndürdü.")

        return ChatResponse(answer=answer, session_id=session_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat endpoint hatası: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Yanıt üretilirken bir hata oluştu. Lütfen tekrar deneyin.",
        )


@app.post(
    "/chat/stream",
    summary="Sohbet Mesajı Gönder (Akış / SSE)",
    tags=["Sohbet"],
    response_description=(
        "Server-Sent Events akışı. Her satır `data: <token>` formatındadır. "
        "Akış `data: [DONE]` ile sonlanır."
    ),
)
async def chat_stream(request: ChatRequest):
    """
    Öğrencinin mesajını alır ve yanıtı **kelime kelime** Server-Sent Events
    (SSE) protokolüyle akışlı olarak iletir.

    **İstemci tarafı kullanımı (JavaScript örneği):**
    ```javascript
    const response = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: 'YKS için nasıl çalışmalıyım?', session_id: 'abc123' })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = decoder.decode(value);
      // "data: token\\n\\n" formatını ayrıştır
      const lines = text.split('\\n').filter(l => l.startsWith('data: '));
      for (const line of lines) {
        const token = line.replace('data: ', '');
        if (token === '[DONE]') break;
        // token'ı UI'a ekle
      }
    }
    ```

    **Not:** `session_id` belirtilmezse otomatik oluşturulur; SSE header'larına
    `X-Session-Id` olarak eklenir.
    """
    session_id = _resolve_session_id(request.session_id)
    logger.info(
        "Stream isteği — session: %s | mesaj: %.80s…",
        session_id,
        request.message,
    )

    return StreamingResponse(
        _token_generator(query=request.message, session_id=session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",        # Nginx proxy buffer'ı devre dışı
            "X-Session-Id": session_id,        # İstemci bu değeri saklayabilir
        },
    )


@app.delete(
    "/session/{session_id}",
    response_model=SessionDeleteResponse,
    summary="Oturum Geçmişini Temizle",
    tags=["Oturum"],
)
async def delete_session(session_id: str):
    """
    Belirtilen `session_id`'ye ait sohbet geçmişini bellekten siler.

    Kullanıcı "Yeni Sohbet" başlattığında veya oturum sonlandığında
    bu endpoint'i çağırarak belleği temizleyin.
    """
    if not session_id or not session_id.strip():
        raise HTTPException(status_code=400, detail="Geçersiz session_id.")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, clear_session_history, session_id)
    logger.info("Oturum silindi: %s", session_id)

    return SessionDeleteResponse(deleted=True, session_id=session_id)
