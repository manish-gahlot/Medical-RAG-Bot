# 🏥 Medical RAG Chatbot

A specialized Retrieval-Augmented Generation (RAG) chatbot designed to answer **medical-related queries** using information retrieved from PDF documents.

Built with **LangChain**, **Pinecone Serverless**, and **Hugging Face** open-source LLMs.

---

## 🚀 Features

- **Automated PDF Ingestion:** Loads and processes all PDFs from a local folder.
- **Smart Text Chunking:** Uses *Recursive Character Text Splitter* for optimized context retrieval.
- **Fast Vector Search:** Powered by **Pinecone** for high-speed similarity matching.
- **Open Source Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2`.
- **Free LLM Inference:** Integrates with `flan-t5-base` via Hugging Face Hub.

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python 3.9+ |
| Framework | LangChain |
| Vector DB | Pinecone Serverless |
| LLM Provider | Hugging Face Hub |
| Embeddings | Sentence Transformers |

---

## 📂 Project Structure

```bash
Medical-RAG-Bot/
├── Data/                  # Place your medical PDFs here
├── .env                   # API keys (not uploaded)
├── .env.example           # Environment variable template
├── main.py                # Main RAG pipeline script
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Medical-RAG-Bot.git
cd Medical-RAG-Bot
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env` File

Add the required API keys:

```ini
PINECONE_API_KEY=your_actual_pinecone_key
HUGGINGFACEHUB_API_TOKEN=your_actual_hf_token
```

### 5. Add Your Data

Create a `Data` folder and paste your medical PDF files into it.

---

## 🏃‍♂️ Usage

Run the main script:

```bash
python main.py
```

> 💡 *First run takes longer due to PDF processing and vector creation in Pinecone.*

---

## 🧠 How It Works

1. **Load:** Scans the `Data/` folder for PDFs.
2. **Split:** Breaks text into 500-character chunks.
3. **Embed:** Converts chunks into vectors using Hugging Face embeddings.
4. **Store:** Uploads embeddings to Pinecone (`medibot` index).
5. **Retrieve:** Fetches the most relevant chunks for any user query.
6. **Generate:** Combines retrieved context + question → sends to `Flan-T5` → generates the answer.

---

## 🤝 Contributing

Pull Requests and suggestions are always welcome!

---
