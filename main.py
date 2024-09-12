
from database_controller import DatabaseController
from query_controller import QueryController

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
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

DatabaseController = DatabaseController(DATABASE, DATA_PATH)
QueryController    = QueryController(DATABASE, LLM_MODEL, QUERY_NUM)

#=============================================================================#

QUERY_PROMPT_TEMPLATE = """

{context}

---

根據以上資料用繁體中文回答問題: {question}
"""

#=============================================================================#

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "使用繁體中文回答問題"}]

st.set_page_config(layout="wide")

#=============================================================================#

def ollama_generator(model_name: str, messages: Dict) -> Generator:
    
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    
    for chunk in stream:
        yield chunk['message']['content']

#=============================================================================#

st.title("RAG demo")

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

    if "更新資料庫" in question:
        prompt = DatabaseController.update_db()

    elif "重設資料庫" in question:
        prompt = DatabaseController.reset_db()

    elif "清除資料庫" in question:
        prompt = DatabaseController.clear_db()

    else:
        results = QueryController.generate_results(question)
        prompt  = QueryController.generate_prompt(question, results, QUERY_PROMPT_TEMPLATE)

    st.session_state.messages.append({"role": "user", "content": prompt})

#-----------------------------------------------------------------------------#

    with st.chat_message("assistant", avatar="🤖"):

        response = st.write_stream(ollama_generator(LLM_MODEL_NAME, st.session_state.messages))

    st.session_state.messages[-1]["content"] = question

    st.session_state.messages.append({"role": "assistant", "content": response})
