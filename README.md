# 🤖 AI Companion

Un assistente AI locale con supporto **RAG** (Retrieval-Augmented Generation) e **TTS** (Text-to-Speech), costruito con Flask, LangChain, Ollama e Kokoro.

---

## 🧠 Stack tecnologico

| Componente | Tecnologia |
|---|---|
| Backend | Flask + Flask-CORS |
| LLM | Ollama (`gemma3:4b`) |
| Embeddings | Ollama (`embeddinggemma:300m`) |
| Vector Store | LangChain `InMemoryVectorStore` |
| TTS | Kokoro ONNX (`if_sara`, italiano) |
| Audio | SoundFile |

---

## 📁 Struttura del progetto

```
Ai_Companion/
├── AI_companion.py       # Backend Flask
├── fronthead.html        # Frontend
├── config.json           # Configurazione modello Kokoro
├── knowledge_base.pdf    # Base di conoscenza
├── vs/
│   └── manga_anime.db    # Vector Store RAG
├── kokoro-v1.0.fp16.onnx # Modello TTS (non incluso)
├── voices-v1.0.bin       # Voci Kokoro (non incluso)
└── kokoro-v1.0.pth       # Weights PyTorch (non incluso)
```

> ⚠️ I file `.onnx`, `.pth` e `.bin` non sono inclusi nel repo per via delle dimensioni. Scaricali separatamente da [Kokoro](https://github.com/thewh1teagle/kokoro-onnx).

---

## 🚀 Avvio

```bash
pip install flask flask-cors langchain-ollama langchain-community kokoro-onnx soundfile
python AI_companion.py
```

Il server parte su `http://localhost:9000`.

---

## 🔌 Endpoint API

### `POST /chat`
Invia un messaggio all'assistente.
```json
{ "message": "Chi è Luffy?" }
```
```json
{ "response": "Monkey D. Luffy è il protagonista di One Piece..." }
```

### `POST /tts`
Converte testo in audio WAV con voce italiana.
```json
{ "text": "Ciao, come posso aiutarti?" }
```
Ritorna un file `audio/wav`.

### `POST /reset`
Azzera la cronologia della conversazione.

### `GET /test`
Health check del server.
```json
{ "status": "ok", "modello": "gemma3:4b" }
```

---

## ⚙️ Come funziona

1. L'utente invia un messaggio via `/chat`
2. Il retriever cerca i documenti più rilevanti nel vector store (`manga_anime.db`)
3. Il contesto recuperato viene iniettato nel system prompt
4. `gemma3:4b` genera la risposta in italiano
5. La risposta può essere sintetizzata vocalmente via `/tts` con Kokoro

---

## 📝 Note

- Il modello risponde **sempre in italiano**
- La cronologia conversazione è **in memoria** (si resetta al riavvio o via `/reset`)
- Il vector store è precaricato da `vs/manga_anime.db`
