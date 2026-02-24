"""
======================================================
  US High School Educational Chatbot
  --------------------------------------------------
  A Retrieval-Augmented Generation (RAG) chatbot that
  answers high school students' questions using a
  local policy knowledge base.

  Tech stack:
    - OpenAI GPT-4o-mini    : language model
    - LangChain             : LLM orchestration
    - ChromaDB              : local vector database
    - OpenAI Embeddings     : text vectorisation

  Usage:
    1. Add your OpenAI API key to a .env file:
         OPENAI_API_KEY=sk-...
    2. Run:  python chatbot.py
======================================================
"""

import os
import logging

# Suppress ChromaDB's broken telemetry client before any imports trigger it
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

from dotenv import load_dotenv

# ----- LangChain imports -----
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Load the OPENAI_API_KEY from the .env file in the same directory
load_dotenv()

# File that contains all educational policy information
POLICIES_FILE = "policies.txt"

# Folder where ChromaDB will persist its local vector database
CHROMA_DB_DIR = "chroma_db"

# OpenAI model to use for chat responses
# gpt-4o-mini: significantly smarter than gpt-3.5-turbo, still very cheap
# Pricing: ~$0.15/1M input tokens, ~$0.60/1M output tokens (vs gpt-4o at $5/$15)
CHAT_MODEL = "gpt-4o-mini"

# Number of relevant document chunks to retrieve per question
TOP_K_RESULTS = 4


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Load and split the policy document
# ─────────────────────────────────────────────────────────────────────────────

def load_policy_documents(filepath: str):
    """
    Reads the policies text file and splits it into smaller chunks.

    Splitting is necessary because language models have a limited context
    window. Smaller chunks also improve retrieval precision — only the
    most relevant passages are sent to the model.

    Args:
        filepath: path to the .txt policies file

    Returns:
        List of LangChain Document objects (text chunks)
    """
    # TextLoader reads the entire file as one document
    loader = TextLoader(filepath, encoding="utf-8")
    raw_docs = loader.load()

    # RecursiveCharacterTextSplitter breaks text at natural boundaries
    # (paragraphs → sentences → words) to stay within chunk_size characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # maximum characters per chunk
        chunk_overlap=100,    # overlap keeps context across chunk boundaries
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[INFO] Loaded '{filepath}' → {len(chunks)} text chunks.")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Build (or load) the vector database
# ─────────────────────────────────────────────────────────────────────────────

def get_vector_store(chunks, persist_dir: str):
    """
    Creates a ChromaDB vector store from the document chunks, or loads the
    existing one if data has already been embedded and stored locally.

    Text embeddings are numerical representations of meaning. ChromaDB stores
    them on disk so we don't re-embed the document every time the app starts.

    Args:
        chunks:      list of Document chunks (from load_policy_documents)
        persist_dir: local folder for ChromaDB storage

    Returns:
        A LangChain Chroma vector store object
    """
    # OpenAI's text-embedding-ada-002 model converts text → 1536-D vectors
    embedding_model = OpenAIEmbeddings()

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        # Database already exists — just load it (saves API calls & time)
        print(f"[INFO] Loading existing vector store from '{persist_dir}'.")
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
    else:
        # First run — embed all chunks and save to disk
        print(f"[INFO] Creating new vector store in '{persist_dir}'...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir,
        )
        print("[INFO] Vector store saved.")

    return vector_store


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Build the conversational RAG chain
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_chain(vector_store):
    """
    Assembles the Retrieval-Augmented Generation (RAG) pipeline:

      User question
           │
           ▼
    Vector store retriever  ← finds the TOP_K most relevant chunks
           │
           ▼
    ChatOpenAI (GPT-3.5)    ← generates an answer grounded in those chunks
           │
           ▼
      Answer to user

    ConversationBufferMemory remembers previous turns so the bot can handle
    follow-up questions (e.g., "Tell me more about that").

    Args:
        vector_store: the ChromaDB vector store

    Returns:
        A LangChain ConversationalRetrievalChain
    """
    # The retriever searches the vector DB for similar chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    # Chat model — temperature=0 keeps answers factual and consistent
    llm = ChatOpenAI(
        model_name=CHAT_MODEL,
        temperature=0,
    )

    # Memory stores the conversation history so the model has context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Combine retriever + LLM + memory into a single chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,  # set True to debug retrieved chunks
        verbose=False,
    )
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Interactive chat loop
# ─────────────────────────────────────────────────────────────────────────────

def run_chatbot(chain):
    """
    Starts the command-line chat interface.

    The loop:
      1. Reads the student's question from the terminal.
      2. Sends it through the RAG chain.
      3. Prints the answer.
      4. Repeats until the student types 'exit' or 'quit'.

    Args:
        chain: the ConversationalRetrievalChain built by build_chat_chain()
    """
    print("\n" + "=" * 60)
    print("  Welcome to the US High School Educational Chatbot!")
    print("  Ask me anything about graduation, college, financial")
    print("  aid, standardized tests, student rights, and more.")
    print("  Type 'exit' or 'quit' to close the chatbot.")
    print("=" * 60 + "\n")

    while True:
        # Get input from the student
        user_input = input("You: ").strip()

        # Ignore empty input
        if not user_input:
            continue

        # Exit commands
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("\nChatbot: Goodbye! Good luck with your studies! 🎓")
            break

        # Send the question to the RAG chain and get the answer
        try:
            response = chain.invoke({"question": user_input})
            answer = response.get("answer", "Sorry, I could not generate an answer.")
            print(f"\nChatbot: {answer}\n")
        except Exception as error:
            print(f"\n[ERROR] {error}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Orchestrates the full startup sequence:
      1. Validate that OPENAI_API_KEY is set.
      2. Load and chunk the policies document.
      3. Build / load the vector database.
      4. Assemble the RAG chain.
      5. Start the interactive chat loop.
    """
    # Guard: make sure the API key is present before doing anything
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY is not set.")
        print("  Create a .env file with: OPENAI_API_KEY=sk-...")
        return

    # Guard: make sure the policies file exists
    if not os.path.exists(POLICIES_FILE):
        print(f"[ERROR] Policies file '{POLICIES_FILE}' not found.")
        print("  Make sure 'policies.txt' is in the same folder as chatbot.py.")
        return

    # Build the pipeline
    chunks = load_policy_documents(POLICIES_FILE)
    vector_store = get_vector_store(chunks, CHROMA_DB_DIR)
    chain = build_chat_chain(vector_store)

    # Start chatting
    run_chatbot(chain)


if __name__ == "__main__":
    main()
