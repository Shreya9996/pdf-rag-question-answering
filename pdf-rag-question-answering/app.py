import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Manthan RAG Assistant",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Prompt template (same as the original script)
# ---------------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """you are a rag agent your work is to acording to provided user question and content
            you should send sent relevent information by anlysing the provided content not give any ramdom answer
            give answer base on provided content only""",
        ),
        ("user", """content : {content}
               question : {question}"""),
    ]
)

MODEL_NAME = "mistral-small-2603"

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history: list of {"role", "content"}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "model" not in st.session_state:
    st.session_state.model = None


@st.cache_resource(show_spinner=False)
def load_model():
    return init_chat_model(MODEL_NAME)


def build_vector_store(pdf_path: str, k: int = 3):
    """Load a PDF, split it, embed it, and return a retriever."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    embedding_model = MistralAIEmbeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        persist_directory="Ai_Project_Chroma_Database",
        embedding=embedding_model,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return vector_store, retriever, len(chunks)


def answer_query(retriever, model, query: str) -> str:
    matched_docs = retriever.invoke(query)
    content = "\n\n".join(doc.page_content for doc in matched_docs)

    final_prompt = PROMPT.invoke({"content": content, "question": query})
    result = model.invoke(final_prompt)
    return result.content


# ---------------------------------------------------------------------------
# Sidebar — document upload & settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Manthan RAG")
    st.caption("Upload a PDF, then ask questions about it.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=3)

    process_clicked = st.button("Process document", type="primary", use_container_width=True)

    if process_clicked:
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Reading, splitting and embedding the document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    vector_store, retriever, n_chunks = build_vector_store(tmp_path, k=top_k)

                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = retriever
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.model = load_model()
                    st.session_state.messages = []  # reset chat on new document

                    os.unlink(tmp_path)

                    st.success(f"Processed '{uploaded_file.name}' into {n_chunks} chunks.")
                except Exception as e:
                    st.error(f"Something went wrong while processing the document: {e}")

    st.divider()
    if st.session_state.processed_file:
        st.info(f"Active document:\n**{st.session_state.processed_file}**")
    else:
        st.caption("No document processed yet.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Ask questions about your document")

if not st.session_state.retriever:
    st.info("👈 Upload a PDF and click **Process document** to get started.")
else:
    # render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question about the document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_query(
                        st.session_state.retriever,
                        st.session_state.model,
                        query,
                    )
                except Exception as e:
                    answer = f"Error while generating the answer: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})