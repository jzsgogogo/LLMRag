# ======================= 文档处理 =======================
import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Model_loader import CHROMA_PERSIST_DIR

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs("docs", exist_ok=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.docs_loaded = False

def load_document(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata['page'] = i + 1
        return docs
    elif ext == 'txt':
        try:
            return TextLoader(file_path).load()
        except:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": file_path})]
    elif ext == 'json':
        return JSONLoader(file_path).load()
    elif ext == 'md':
        return UnstructuredMarkdownLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def process_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = embeddings.embed_documents([d.page_content for d in docs_split])
    return docs_split, vectors
import shutil

if os.path.exists("docs"):
    shutil.rmtree("docs")
os.makedirs("docs", exist_ok=True)