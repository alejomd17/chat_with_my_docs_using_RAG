import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

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
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Streamlit Settings
st.set_page_config(page_title="Chat with documents 📚",page_icon="📚")
st.title("Chat with documents 📚")

# model_class = "hf_hub" # @param ["hf_hub","ollama","openai","groq"]
model_class = "groq" # @param ["hf_hub","ollama","openai","groq"]

# Model providers
# def model_hf_hub(model = "meta-llama/Meta-Llama-3-8B-Instruct", temperature = 0.1):
# def model_hf_hub(model = "meta-llama/Llama-2-7b-chat-hf", temperature = 0.1):
def model_hf_hub(model = "mistralai/Mistral-7B-Instruct-v0.1", temperature = 0.1):
  llm = HuggingFaceEndpoint(
      repo_id=model,
      task="text-generation",
      huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),  # From .env
      temperature = temperature,
      max_new_tokens = 512,
      return_full_text = False
  )
  return llm
  

def model_openai(model = "gpt-4o-mini", temperature = 0.1):
  llm = ChatOpenAI(model = model, temperature = temperature)
  return llm

# def model_ollama(model = "phi3", temperature = 0.1):
def model_ollama(model = "llama3", temperature = 0.1):
  llm = ChatOllama(model = model, temperature = temperature)
  return llm

def model_groq(model="llama3-70b-8192",temperature=0.1):
  llm = ChatGroq(model=model, temperature=temperature, max_tokens=None, timeout=None, max_retries=2)
  return llm
    
# Indexing and retrieval
def config_retriever(uploads):
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
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 1000,
                    chunk_overlap = 200)
    splits = text_splitter.split_documents(docs)

    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-m3")
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Store
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    # Retrieve
    retriever = vectorstore.as_retriever(search_type = "mmr",
                                         search_kwargs={'k':3,'fetch_k':4})
    
    return retriever

def config_rag_chain(model_class, retriever):
    # Loading the LLM
    if model_class == 'hf_hub':
        llm = model_hf_hub()
    elif model_class == 'openai':
        llm = model_openai()
    elif model_class == 'ollama':
        llm = model_ollama()
    elif model_class == 'groq':
        llm = model_groq()

    # Prompt definition
    if model_class.startswith("hf"):
        token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>","<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else: 
        token_s, token_e = "",""
    
    # query -> retriever
    # (query, chat_history) -> LLM -> reformulated query -> retriever
    # Talk about company 1 ?
    # When was it founded ?
    # Contextualization prompt
    context_q_system_promt = "Given the following chat histroy and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    context_q_system_promt = token_s + context_q_system_promt
    context_q_user_prompt = "Question: {input}" + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
       [
          ("system", context_q_system_promt),
          MessagesPlaceholder("chat_history"),
          ("human", context_q_user_prompt)
       ]
    )

    # Chain for contextualization
    history_aware_retriever = create_history_aware_retriever(llm = llm,
                                                             retriever = retriever,
                                                             prompt = context_q_prompt)
    
    # Q&A Prompt
    qa_prompt_template = """You are a helpful virtual assistant answering general questions.
    Use the following bits of retrieved context to answer the question.
    If you don't know the answer, just say tou don't know. Keep your answer concise.
    Answer in English. \n\n
    Question: {input} \n
    Context: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    # Configure LLM and Chain for Q&A
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain


uploads = st.sidebar.file_uploader(
  label = "Upload files", type = ["pdf"],
  accept_multiple_files = True  
)
if not uploads:
  st.info("Please send some files to continue")
  st.stop()

if "chat_history" not in st.session_state:
   st.session_state.chat_history = [
        AIMessage(content = "Hi, I'm your virtual assistant! How can I help you?")
   ]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)       

# we use time to measure how long it took for generation
start = time.time()
user_query = st.chat_input("Enter your message here...")

if user_query is not None and user_query != "" and uploads is not None:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        if st.session_state.docs_list != uploads:
            print(uploads)
            st.session_state.docs_list = uploads
            st.session_state.retriever = config_retriever(uploads)

        rag_chain = config_rag_chain(model_class, st.session_state.retriever)

        result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})

        resp = result['answer']
        st.write(resp)

        # Show the source
        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page','Page not specified')

            ref = f":link: Soruce {idx}: *{file} - p, {page}*"

            print(ref)
            with st.popover(ref):
                st.caption(doc.page_content)

    st.session_state.chat_history.append(AIMessage(content=resp))

end = time.time()
print("Time: ", end - start)