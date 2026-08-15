**Retrieval-Augmented Generation (RAG) based PDF Question Answering application** built using **LangChain, Mistral AI, ChromaDB, and Python**.

The application loads information from a PDF document, splits it into smaller chunks, converts those chunks into vector embeddings, stores them in ChromaDB, and retrieves the most relevant information whenever the user asks a question. The retrieved context is then provided to a Mistral LLM to generate a relevant and document-grounded answer.

---

## 🚀 Features

* 📄 Load and process PDF documents
* ✂️ Split documents into smaller chunks
* 🧠 Generate embeddings using Mistral AI
* 🗄️ Store document embeddings in ChromaDB
* 🔎 Perform similarity-based document retrieval
* 🤖 Generate answers using Mistral Small
* 📚 Answer questions based only on retrieved document content
* 💻 Interactive command-line question answering
* 🔐 Environment variable support for API keys

---

## 🏗️ RAG Architecture

```text
                PDF Document
                     │
                     ▼
              PyPDFLoader
                     │
                     ▼
             Document Chunks
                     │
                     ▼
        RecursiveCharacterTextSplitter
          Chunk Size: 1000
          Overlap: 200
                     │
                     ▼
          Mistral AI Embeddings
                     │
                     ▼
                ChromaDB
                     │
                     ▼
               Retriever
                Top K = 3
                     │
             User Question
                     │
                     ▼
           Relevant Documents
                     │
                     ▼
              RAG Prompt
                     │
                     ▼
           Mistral Small 2603
                     │
                     ▼
              Final Answer
```

---

## 🛠️ Technologies Used

| Technology                     | Purpose                            |
| ------------------------------ | ---------------------------------- |
| Python                         | Core programming language          |
| LangChain                      | RAG pipeline and LLM orchestration |
| PyPDFLoader                    | Loading PDF documents              |
| RecursiveCharacterTextSplitter | Document chunking                  |
| Mistral AI Embeddings          | Creating vector embeddings         |
| ChromaDB                       | Vector database                    |
| Mistral Small 2603             | Large Language Model               |
| python-dotenv                  | Environment variable management    |

---

## 📂 Project Structure

```text
mistral-rag-pdf-agent/
│
├── Rag.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
└── Manthan-Project-Document.pdf
```

> The ChromaDB directory is generated locally when the application runs and should not be committed to GitHub.

---

## ⚙️ How It Works

### 1. Load the PDF

The application uses `PyPDFLoader` to read the PDF document.

```python
data = PyPDFLoader("Manthan-Project-Document.pdf")
docs = data.load()
```

### 2. Split the Document

The document is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

This helps the retrieval system find relevant portions of the document efficiently.

### 3. Generate Embeddings

Mistral AI embeddings are used to convert text chunks into numerical vector representations.

```python
embedding_model = MistralAIEmbeddings()
```

### 4. Store Embeddings

The embeddings are stored in ChromaDB.

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    persist_directory="Ai_Project_Chroma_Database",
    embedding=embedding_model
)
```

### 5. Retrieve Relevant Information

When the user asks a question, the retriever searches for the most relevant chunks.

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

### 6. Generate the Answer

The retrieved context is passed to the Mistral model along with the user's question.

```text
Retrieved Context + User Question
                ↓
          Mistral Small
                ↓
           Final Answer
```

---
<img width="1365" height="665" alt="image" src="https://github.com/user-attachments/assets/85cec0f3-3d7b-4f19-9bbf-d782ed070631" />


<img width="1354" height="625" alt="image" src="https://github.com/user-attachments/assets/ac71dbe7-cc18-49fa-9fca-e92f828b732d" />



## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

Never upload your actual `.env` file or API key to GitHub.

You can create a `.env.example` file:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Shreya9996/pdf-rag-question-answering.git
```

### 2. Navigate to the Project

```bash
cd mistral-rag-pdf-agent
```

### 3. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

Create a `.env` file and add:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

### 6. Run the Application

```bash
python Rag.py
```

---

## 💬 Example

After running the application:

```text
Rag Application is Created!

Ask Question: What is this project?

Agent Answer:
Manthan is an AI-powered project planning system...
```

Another example:

```text
Ask Question: What technologies are required for this project?

Agent Answer:
The project uses React, Vite, TypeScript, Tailwind CSS...
```

You can also ask questions such as:

```text
What is the project objective?
Who are the project members?
What technologies are used?
What are the key features?
What is the system architecture?
```

---

## 🎯 Chunking Configuration

The current document splitting configuration is:

```text
Chunk Size   : 1000
Chunk Overlap: 200
```

These values provide a good starting point for retrieving relevant information from project documentation.

The retrieval configuration is:

```text
Search Type: Similarity
Top K      : 3
```

The values can be adjusted depending on document size and retrieval quality.

---

## 🔮 Future Improvements

* [ ] Add conversational memory
* [ ] Support follow-up questions
* [ ] Implement Conversational RAG
* [ ] Add Streamlit web interface
* [ ] Add PDF upload functionality
* [ ] Support multiple PDF documents
* [ ] Add source/page references to answers
* [ ] Add metadata-based filtering
* [ ] Improve retrieval using MMR
* [ ] Add reranking for better retrieval
* [ ] Deploy the application as a web service

---

## 📌 Limitations

The current version is a basic RAG implementation.

* It does not maintain conversation history.
* Each user question is processed independently.
* The quality of answers depends on document chunking and retrieval.
* The model can only answer effectively when relevant information is retrieved from the document.

---

## 👩‍💻 Author

**Shreya Patil**

B.Tech Computer Science Engineering Student
Interested in **Data Science, Artificial Intelligence, Machine Learning, Generative AI, and Automation**.

---

## ⭐ Acknowledgement

This project was developed as a practical implementation of **Retrieval-Augmented Generation (RAG)** using the LangChain ecosystem, Mistral AI, and ChromaDB.

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and research purposes.
