
from database_controller import populate_database, clear_database, calculate_existing_ids
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from query_controller import generate_results, generate_prompt
from langchain_chroma import Chroma
from typing import Dict, Generator
import streamlit as st
import ollama

#=============================================================================#

LLM_MODEL_NAME       = "gemma2:2b"
EMBEDDING_MODEL_NAME = "all-minilm"

QUERY_NUM       = 5
DATA_PATH       = "data"
CHROMA_PATH     = "chroma"

LLM_MODEL       = Ollama(model=LLM_MODEL_NAME)
EMBEDDING_MODEL = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

# 初始化Chroma向量存儲
DATABASE = Chroma(
    persist_directory  = CHROMA_PATH, 
    embedding_function = EMBEDDING_MODEL
    )

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "使用繁體中文回答問題"}]

#=============================================================================#

PROMPT_TEMPLATE = """

{context}

---

根據以上資料用繁體中文回答問題: {question}
"""

#=============================================================================#

def ollama_generator(model_name: str, messages: Dict) -> Generator:
    
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    
    for chunk in stream:
        yield chunk['message']['content']

#-----------------------------------------------------------------------------#

def update_database():

    st.session_state.messages.append({"role": "assistant", "content": "更新資料庫"})

    existing_ids = calculate_existing_ids(DATABASE)
    st.session_state.messages.append({"role": "assistant", "content": f"Number of existing documents in DB: {len(existing_ids)}"})

    new_chunks = populate_database(EMBEDDING_MODEL, DATA_PATH, DATABASE)

    if len(new_chunks):
        st.session_state.messages.append({"role": "assistant", "content": f"Adding new documents: {len(new_chunks)}"})

    else:
        st.session_state.messages.append({"role": "assistant", "content": "No new documents to add"})

#-----------------------------------------------------------------------------#

def reset_database():

    st.session_state.messages.append({"role": "assistant", "content": "重置資料庫"})

    delete_ids = calculate_existing_ids(DATABASE)
    clear_database(delete_ids, DATABASE)

    existing_ids = calculate_existing_ids(DATABASE)
    st.session_state.messages.append({"role": "assistant", "content": f"Number of existing documents in DB: {len(existing_ids)}"})

    new_chunks = populate_database(EMBEDDING_MODEL, DATA_PATH, DATABASE)

    if len(new_chunks):
        st.session_state.messages.append({"role": "assistant", "content": f"Adding new documents: {len(new_chunks)}"})

    else:
        st.session_state.messages.append({"role": "assistant", "content": "No new documents to add"})

#-----------------------------------------------------------------------------#

def clear_database():

    st.session_state.messages.append({"role": "assistant", "content": "清空資料庫"})

    delete_ids = calculate_existing_ids(DATABASE)
    clear_database(delete_ids, DATABASE)

    st.session_state.messages.append({"role": "assistant", "content": "Clearing Database"})

#=============================================================================#

st.title("RAG demo")

#-----------------------------------------------------------------------------#

with st.sidebar:
    if st.button("更新資料庫"):
        update_database()
        
    if st.button("重置資料庫"):
        reset_database()
        
    if st.button("清空資料庫"):
        clear_database()

#-----------------------------------------------------------------------------#

for message in st.session_state.messages[1:]:

    if message["role"] == "user":
        with st.chat_message("user", avatar="🦖"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])

#-----------------------------------------------------------------------------#

if question := st.chat_input("How could I help you?"):

    with st.chat_message("user", avatar="🦖"):
        st.markdown(question)

    results = generate_results(question, QUERY_NUM, DATABASE)
    prompt  = generate_prompt(question, results, PROMPT_TEMPLATE)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):

        response = st.write_stream(ollama_generator(LLM_MODEL_NAME, st.session_state.messages))

    st.session_state.messages[-1]["content"] = question

    st.session_state.messages.append({"role": "assistant", "content": response})
