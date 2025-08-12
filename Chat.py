import streamlit as st
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
import faiss
import tempfile
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
# Streamlit Settings
st.set_page_config(page_title="Chat with documents 📚",page_icon="📚")
st.title("Chat with documents 📚")


def model_hf_hub(model = "meta-llama/Meta-Llama-3-8B-Instruct", temperature = 0.1):
  llm = HuggingFaceEndpoint(
      repo_id=model,
      temperature=temperature,
      max_new_tokens = 512,
      return_full_text = False,
  )
  return llm

# def model_openai(model = "gpt-4o-mini", temperature = 0.1):
#   llm = ChatOpenAI(model = model, temperature = temperature)
#   return llm

def model_ollama(model = "phi3", temperature = 0.1):
  llm = ChatOllama(model = model, temperature = temperature)
  return llm

def model_groq(model="llama3-70b-8192",temperature=0.1):
  llm = ChatGroq(model=model, temperature=temperature, max_tokens=None, timeout=None, max_retries=2)
  return llm

uploads = st.sidebar.file_uploader(
  label = "Upload files", types = ["pdf"],
  accept_multiple_files = True  
)
if not uploads:
  st.info("Please send some files to continue")
  st.stop()

# Indexing and retrieval
def config_retrieval(uploads):
    # Load
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    
    # Split
    text_splitter = RecursiveCharacterTextSplitter(hunk_size = 1000, chunk_overlap = 200)
    splits = text_splitter.split_documents(docs)

    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-m3")

    # Store
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    # Retrieve
    retriever = vectorstore.as_retriever(search_type = "mmr",
                                         search_kwargs={'k':3,'fetch_k':4})
    
    return retriever