# 🧠 RAG-Based AI Chatbot with Source Document Insight

This project is a **Retrieval-Augmented Generation (RAG)** powered AI chatbot that uses Hugging Face models and LangChain to answer user queries based on content extracted from `.docx` files. It features a web-based chat UI with enhanced capabilities such as source citation, keyword highlighting, and document downloads.

---

## 🚀 Features

- RAG pipeline using `Mistral-7B-Instruct` (via HuggingFace)
- Pre-indexing of `.docx` documents using FAISS
- Smart chat UI with:
  - Previous answer browsing
  - Keyword highlighting
  - Source-based content display
  - Download buttons (extracted content or full document)
- Spelling correction using TextBlob
- Semantic matching using Levenshtein and cosine similarity

---

## 🗂 Project Structure
├── rag_hf_distilbart_preindex.py # Indexes documents and builds vectorstore
├── rag_hf_distilbart_only_query_server.py# Flask app with RAG query backend
├── app_with_answer.js # Chat UI logic for sending/receiving queries
├── templates/
│ ├── base.html # Flask-rendered chat interface
├── static/
│ ├── images/ # Icons for UI buttons
│ ├── style.css # Styling for chatbox
│ ├── app.js # JS loaded by base.html
├── documents/ # Folder containing .docx source files
├── vectorstore/ # Saved FAISS index and metadata
│ ├── index.faiss
│ ├── index.pkl
│ └── index_metadata.json

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot

2. Install Python dependencies

pip install -r requirements.txt

pip install flask langchain transformers sentence-transformers spacy textblob python-docx faiss-cpu python-Levenshtein
python -m spacy download en_core_web_sm

3. Index your documents
Place your .docx files inside the documents/ folder and run:
python rag_hf_distilbart_preindex.py

This will create the FAISS vectorstore and update the metadata.

4. Start the Flask server
python rag_hf_distilbart_only_query_server.py

The server runs at: http://127.0.0.1:5000


🌐 Using the Web Chat UI
Open your browser and navigate to: http://127.0.0.1:5000

Enter a query in the input field.

Click Send to get answers.

Click Prev to see past responses.

Click Refresh to clear the chat.

View sources and download:

Extracted content (.txt)

Original document file (.docx)

Additional content via “More Info” button (if available)

🧠 How It Works
Backend
Indexing (preindex.py): Loads .docx files → extracts content → splits into chunks → computes embeddings → saves FAISS index.

Querying (query_server.py): Accepts user query → corrects spelling → lemmatizes → retrieves relevant chunks → generates response with mistralai/Mistral-7B-Instruct.

Frontend
app_with_answer.js: Manages chat interaction, response display, keyword highlighting, and file downloads.

base.html: Chat interface loaded via Flask.