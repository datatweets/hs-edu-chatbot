"""
======================================================
  US High School Educational Chatbot — Web UI
  --------------------------------------------------
  Flask server that wraps the RAG pipeline from
  chatbot.py and exposes two endpoints:

    GET  /          → serves the chat HTML page
    POST /chat      → receives a question, returns
                      the answer as JSON

  Run:
    python app.py
  Then open:  http://127.0.0.1:5000
======================================================
"""

import os
import logging

# Suppress ChromaDB telemetry warnings before any imports trigger it
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

# Reuse the exact same pipeline functions from chatbot.py
from chatbot import load_policy_documents, get_vector_store, build_chat_chain

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

POLICIES_FILE = "policies.txt"
CHROMA_DB_DIR = "chroma_db"

# ─────────────────────────────────────────────────────────────────────────────
# Initialise Flask and the RAG chain (done once at startup)
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Build the chain when the server starts so the first request isn't slow
print("[INFO] Initialising RAG pipeline...")
_chunks = load_policy_documents(POLICIES_FILE)
_store  = get_vector_store(_chunks, CHROMA_DB_DIR)
chain   = build_chat_chain(_store)
print("[INFO] Chatbot ready. Visit http://127.0.0.1:5000")

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Receive a JSON body: { "message": "student question" }
    Return a JSON body:  { "answer": "chatbot reply" }
    """
    data = request.get_json(silent=True)

    if not data or not data.get("message", "").strip():
        return jsonify({"error": "No message provided."}), 400

    user_message = data["message"].strip()

    try:
        response = chain.invoke({"question": user_message})
        answer   = response.get("answer", "Sorry, I could not generate an answer.")
        return jsonify({"answer": answer})
    except Exception as err:
        return jsonify({"error": str(err)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY is not set. Add it to your .env file.")
    else:
        # debug=False for a clean production-like experience
        app.run(host="127.0.0.1", port=5000, debug=False)
