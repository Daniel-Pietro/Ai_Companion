from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from kokoro_onnx import Kokoro
import soundfile as sf
import io
import os

app = Flask(__name__)
CORS(app)

# --- Carica il Vector Store (RAG) ---
embeddings = OllamaEmbeddings(model="embeddinggemma:300m")
vs = InMemoryVectorStore(embedding=embeddings)
vs.load("./vs/manga_anime.db", embeddings)
retriever = vs.as_retriever(search_kwargs={"k": 3})

# --- Carica il modello LLM ---
lcmodel = ChatOllama(model="gemma3:4b", temperature=0, reasoning=False)

# --- Carica Kokoro TTS ---
kokoro = Kokoro("kokoro-v1.0.fp16.onnx", "voices-v1.0.bin")

# --- Prompt di sistema ---
SYSTEM_PROMPT = """Sei un assistente AI utile e cordiale.
Rispondi sempre in italiano.
Usa le seguenti informazioni di contesto per rispondere alla domanda dell'utente:

{context}

Se il contesto non contiene informazioni utili, rispondi comunque al meglio delle tue conoscenze."""

# --- Cronologia conversazione ---
chat_history = []

def get_context(query):
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Messaggio vuoto"}), 400

    context = get_context(user_message)

    messages = [("system", SYSTEM_PROMPT.format(context=context))]
    for msg in chat_history:
        messages.append(msg)
    messages.append(("human", user_message))

    response = lcmodel.invoke(messages)
    ai_reply = response.content

    chat_history.append(("human", user_message))
    chat_history.append(("ai", ai_reply))

    return jsonify({"response": ai_reply})

@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "")

    samples, sample_rate = kokoro.create(
        text,
        voice="if_sara",
        speed=0.85,
        lang="it"
    )

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format='WAV')
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")

@app.route("/reset", methods=["POST"])
def reset():
    chat_history.clear()
    return jsonify({"status": "cronologia azzerata"})

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "modello": "gemma3:4b"})

if __name__ == "__main__":
    os.makedirs("./generated", exist_ok=True)
    app.run(port=9000, debug=False)