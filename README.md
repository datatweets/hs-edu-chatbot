# EduBot — US High School Educational Chatbot

[![GitHub](https://img.shields.io/badge/GitHub-hs--edu--chatbot-1a3a6b?logo=github)](https://github.com/datatweets/hs-edu-chatbot)

A **Retrieval-Augmented Generation (RAG)** chatbot that answers US high school
students' questions about graduation, college admissions, financial aid,
standardized tests, student rights, career planning, and more — powered by
OpenAI and LangChain, with a local ChromaDB vector database.

The project has **two ways to use it**:

| Mode | How to run | Best for |
|---|---|---|
| **Terminal chatbot** | `python chatbot.py` | Quick testing, scripting |
| **Web UI** | `python app.py` → browser | Real students, demos |

---

## Table of Contents

- [EduBot — US High School Educational Chatbot](#edubot--us-high-school-educational-chatbot)
  - [Table of Contents](#table-of-contents)
  - [1. How it works](#1-how-it-works)
  - [2. Project structure](#2-project-structure)
  - [3. Prerequisites](#3-prerequisites)
    - [Python 3.9 or newer](#python-39-or-newer)
    - [pip (Python package manager)](#pip-python-package-manager)
  - [4. Step-by-step setup](#4-step-by-step-setup)
    - [4.1 Download or clone the project](#41-download-or-clone-the-project)
    - [4.2 Create a virtual environment](#42-create-a-virtual-environment)
    - [4.3 Activate the virtual environment](#43-activate-the-virtual-environment)
    - [4.4 Install dependencies](#44-install-dependencies)
    - [4.5 Get an OpenAI API key](#45-get-an-openai-api-key)
    - [4.6 Create the .env file](#46-create-the-env-file)
  - [5. Running the terminal chatbot](#5-running-the-terminal-chatbot)
  - [6. Running the web UI](#6-running-the-web-ui)
  - [7. First run explained](#7-first-run-explained)
  - [8. Sample questions to try](#8-sample-questions-to-try)
  - [9. Updating the knowledge base](#9-updating-the-knowledge-base)
  - [10. Project files explained](#10-project-files-explained)
  - [11. Troubleshooting](#11-troubleshooting)
  - [12. Cost estimate](#12-cost-estimate)

---

## 1. How it works

```
policies.txt
(your knowledge base)
      │
      ▼
 Split into ~25 small chunks
      │
      ▼
 OpenAI Embeddings convert each chunk into a vector
      │
      ▼
 ChromaDB stores vectors locally in  chroma_db/  folder
      │                    (this only happens ONCE on first run)
      ▼
 Student asks a question
      │
      ▼
 ChromaDB finds the 4 most relevant chunks (semantic search)
      │
      ▼
 GPT-4o-mini reads those chunks + conversation history
      │
      ▼
 Answer is returned to the student
```

**Key concepts:**

- **RAG (Retrieval-Augmented Generation)**: the model only answers based on
  the provided document — it does not make things up.
- **Vector database**: text is stored as numbers (vectors) so the system can
  find *meaning*, not just keywords.
- **Conversation memory**: previous messages are remembered, so follow-up
  questions like *"tell me more about that"* work naturally.

---

## 2. Project structure

```
OpenAI-chatbot/
│
├── chatbot.py           # Terminal chatbot — full RAG pipeline
├── app.py               # Flask web server — wraps chatbot.py for the browser
│
├── templates/
│   └── index.html       # Chat UI (HTML + CSS + JavaScript, no frameworks)
│
├── policies.txt         # Knowledge base — all educational content
├── sample_questions.txt # 20 scenario-based test questions
│
├── requirements.txt     # Python package dependencies
├── .env                 # YOUR secret API key (never share this file)
├── .env.example         # Safe template showing what .env should look like
├── .gitignore           # Prevents secrets and DB from being committed
│
└── chroma_db/           # Auto-created on first run — local vector database
                         # Safe to delete — rebuilt automatically on next run
```

---

## 3. Prerequisites

Before starting, make sure you have the following installed on your computer.

### Python 3.9 or newer

Open your terminal and type:

```bash
python --version
# or, on some systems:
python3 --version
```

You should see something like `Python 3.11.4`.  
If Python is not installed, download it from **https://www.python.org/downloads/**  
✅ During installation on Windows, check **"Add Python to PATH"**.

### pip (Python package manager)

pip comes bundled with Python. Verify it works:

```bash
pip --version
```

---

## 4. Step-by-step setup

### 4.1 Download or clone the project

**Option A — Download ZIP** (no Git required):
1. Download the project as a ZIP file.
2. Unzip it to a folder, e.g. `Documents/OpenAI-chatbot`.

**Option B — Git clone**:
```bash
git clone https://github.com/datatweets/hs-edu-chatbot.git
cd hs-edu-chatbot
```

Then open a terminal and navigate into the project folder:

```bash
# macOS / Linux
cd ~/Documents/OpenAI-chatbot

# Windows
cd C:\Users\YourName\Documents\OpenAI-chatbot
```

---

### 4.2 Create a virtual environment

A virtual environment is an isolated Python workspace. It keeps the packages
for this project separate from everything else on your computer.

```bash
# macOS / Linux
python3 -m venv .venv

# Windows
python -m venv .venv
```

You will see a new folder called `.venv` appear in your project directory.
This folder holds a private copy of Python and all the packages we install.

> **Why bother?**  
> Without a virtual environment, packages install globally and can conflict
> with other projects. It is considered best practice to always use one.

---

### 4.3 Activate the virtual environment

You must activate the environment **every time** you open a new terminal.

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

After activation your terminal prompt will change to show `(.venv)` at the
start — this confirms the environment is active:

```
(.venv) user@computer OpenAI-chatbot %
```

To deactivate later (when you are done):
```bash
deactivate
```

---

### 4.4 Install dependencies

With the virtual environment active, install all required packages:

```bash
pip install -r requirements.txt
```

This will download and install:

| Package | Purpose |
|---|---|
| `langchain` | Orchestrates the RAG pipeline |
| `langchain-community` | Document loaders, vector store wrappers |
| `langchain-openai` | Connects LangChain to OpenAI models |
| `openai` | Official OpenAI Python SDK |
| `chromadb` | Local vector database (saved to disk) |
| `python-dotenv` | Loads the API key from the `.env` file |
| `flask` | Lightweight web server for the browser UI |

Installation takes 1–3 minutes. You will see a lot of lines scrolling —
that is normal.

---

### 4.5 Get an OpenAI API key

The chatbot uses OpenAI's models, which require an API key.

1. Go to **https://platform.openai.com/api-keys**
2. Sign in or create a free account.
3. Click **"Create new secret key"**.
4. Copy the key — it starts with `sk-` and is shown **only once**.

> **Cost note**: The model used is `gpt-4o-mini`, which is extremely
> affordable. A full session of 50 questions typically costs less than
> **$0.01** (one cent). New accounts also receive free trial credits.

---

### 4.6 Create the .env file

The `.env` file stores your API key securely on your computer.
It is listed in `.gitignore` so it will never be accidentally uploaded online.

**Step 1** — Copy the example template:

```bash
# macOS / Linux
cp .env.example .env

# Windows (Command Prompt)
copy .env.example .env
```

**Step 2** — Open `.env` in any text editor (Notepad, VS Code, etc.)
and replace the placeholder with your real key:

```
OPENAI_API_KEY=sk-your-actual-key-goes-here
ANONYMIZED_TELEMETRY=False
```

Save the file. Do not add quotes around the key.

---

## 5. Running the terminal chatbot

Make sure your virtual environment is **active** (you see `(.venv)` in the
terminal prompt), then run:

```bash
python chatbot.py
```

**First run output** (normal):
```
[INFO] Loaded 'policies.txt' → 25 text chunks.
[INFO] Creating new vector store in 'chroma_db'...
[INFO] Vector store saved.

============================================================
  Welcome to the US High School Educational Chatbot!
  ...
============================================================

You:
```

**Subsequent runs** (faster — no re-embedding):
```
[INFO] Loaded 'policies.txt' → 25 text chunks.
[INFO] Loading existing vector store from 'chroma_db'.

You:
```

Type your question and press **Enter**. Type `exit` or `quit` to close.

---

## 6. Running the web UI

Make sure your virtual environment is **active**, then run:

```bash
python app.py
```

You will see:
```
[INFO] Initialising RAG pipeline...
[INFO] Loaded 'policies.txt' → 25 text chunks.
[INFO] Chatbot ready. Visit http://127.0.0.1:5000
```

Open your browser and go to: **http://127.0.0.1:5000**

The chat interface will load with:
- Quick-question chips along the top
- A chat window with conversation bubbles
- A text box to type your question (press Enter to send, Shift+Enter for newline)

To stop the server, press `Ctrl + C` in the terminal.

---

## 7. First run explained

On the very first run, the program reads `policies.txt`, splits it into
~25 text chunks, and sends each chunk to OpenAI to be converted into a
vector (a list of numbers representing its meaning).

These vectors are saved permanently to the `chroma_db/` folder.
**This only happens once** — every subsequent run loads them instantly
from disk without calling OpenAI, saving time and money.

If you ever edit `policies.txt`, delete the `chroma_db/` folder so the
database is rebuilt with the updated content:

```bash
rm -rf chroma_db      # macOS / Linux
rmdir /s /q chroma_db # Windows
```

---

## 8. Sample questions to try

Open `sample_questions.txt` for 20 challenging, scenario-based questions.
Here are a few quick ones to start with:

```
How many credits do I need to graduate?
What is the difference between the SAT and ACT?
How does FAFSA work and when should I apply?
I have ADHD — what accommodations can I get at school?
What is the community college transfer route to a 4-year university?
I got rejected from all my colleges. What are my options now?
Can I still get into college if my GPA is low?
What is the National Merit Scholarship?
```

Follow-up questions work naturally:
```
You: What is Early Decision?
Chatbot: [explains ED]
You: How is that different from Early Action?
Chatbot: [compares both using conversation history]
```

---

## 9. Updating the knowledge base

All answers come from `policies.txt`. To add, change, or remove information:

1. Open `policies.txt` in any text editor.
2. Make your changes and save.
3. Delete the `chroma_db/` folder (so the old vectors are wiped):
   ```bash
   rm -rf chroma_db
   ```
4. Run the chatbot again — it will re-embed the updated document.

---

## 10. Project files explained

| File | What it does |
|---|---|
| `chatbot.py` | The full RAG pipeline. Loads the document, builds the vector DB, sets up the LLM chain, and runs the terminal chat loop. |
| `app.py` | A Flask web server that imports the pipeline from `chatbot.py` and exposes it via a `/chat` API endpoint, serving `index.html` from the `templates/` folder. |
| `templates/index.html` | The entire browser UI — chat bubbles, typing animation, quick chips — written in plain HTML, CSS, and JavaScript with no external frameworks. |
| `policies.txt` | The knowledge base. Contains 10 sections covering graduation, tests, admissions, financial aid, academic programs, student rights, careers, mental health, extracurriculars, and FAQs. |
| `sample_questions.txt` | 20 hard, scenario-based questions designed to test every section of the knowledge base. |
| `requirements.txt` | Lists all Python packages needed. Install with `pip install -r requirements.txt`. |
| `.env` | Stores your `OPENAI_API_KEY`. Never share or commit this file. |
| `.env.example` | A safe template you copy to create `.env`. |
| `.gitignore` | Tells Git to ignore `.env`, `chroma_db/`, and `__pycache__`. |
| `chroma_db/` | Auto-created local vector database. Safe to delete and rebuild. |

---

## 11. Troubleshooting

**`OPENAI_API_KEY` not found / AuthenticationError**
> Make sure `.env` exists in the project folder, the key starts with `sk-`,
> and there are no spaces around the `=` sign.

**`ModuleNotFoundError: No module named 'langchain'`**
> Your virtual environment is not active. Run:
> `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate` (Windows)
> then try again.

**`python: command not found`**
> Try `python3` instead of `python`, or reinstall Python and ensure it is
> added to your system PATH.

**Port 5000 already in use (web UI)**
> Another app is using port 5000. Change the port in the last line of
> `app.py`:
> ```python
> app.run(host="127.0.0.1", port=5001, debug=False)
> ```
> Then visit `http://127.0.0.1:5001`.

**Answers seem wrong or outdated**
> The knowledge comes entirely from `policies.txt`. Edit the file, delete
> `chroma_db/`, and rerun to rebuild with corrected content.

**`Failed to send telemetry event` warnings in terminal**
> These are a known bug in ChromaDB's internal telemetry system and are
> completely harmless. The fix is already applied in `chatbot.py`. If they
> still appear, make sure `ANONYMIZED_TELEMETRY=False` is in your `.env`.

---

## 12. Cost estimate

All OpenAI usage is billed per token (roughly 1 token ≈ 0.75 words).

| Action | Approximate cost |
|---|---|
| First-run embedding of `policies.txt` | ~$0.0001 (less than a fraction of a cent) |
| One question + answer (gpt-4o-mini) | ~$0.0002 |
| 100 questions in one session | ~$0.02 |
| Typical student study session | < $0.01 |

New OpenAI accounts receive free trial credits that will cover hundreds of
sessions before any billing begins.

---

> Answers are based on the content of `policies.txt` which was compiled from
> publicly available US educational resources. Always confirm important
> academic or financial decisions with your school counselor.

---

**Repository:** https://github.com/datatweets/hs-edu-chatbot

