
from langchain_community.embeddings.ollama import OllamaEmbeddings
from database_controller import DatabaseController
from query_controller import QueryController
from langchain_chroma import Chroma
import streamlit as st

#=============================================================================#

LLM_MODEL       = "gemma2:2b"
EMBEDDING_MODEL = "all-minilm"

QUERY_NUM   = 5
DATA_PATH   = "data"
CHROMA_PATH = "chroma"

#=============================================================================#

# 初始化Chroma向量存儲
DATABASE = Chroma(
    persist_directory  = CHROMA_PATH, 
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    )

DatabaseController = DatabaseController(DATABASE, DATA_PATH)

QueryController    = QueryController(DATABASE, LLM_MODEL, QUERY_NUM)

#=============================================================================#

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "使用繁體中文回答問題"}]

st.set_page_config(layout="wide")

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

#-----------------------------------------------------------------------------#

    if "更新資料庫" in question:
        prompt = DatabaseController.update_db()

    elif "重置資料庫" in question:
        prompt = DatabaseController.reset_db()

    elif "清除資料庫" in question:
        prompt = DatabaseController.clear_db()

    else:
        results = QueryController.generate_results(question)
        prompt  = QueryController.generate_prompt(question, results)

    st.session_state.messages.append({"role": "user", "content": prompt})

#-----------------------------------------------------------------------------#

    with st.chat_message("assistant", avatar="🤖"):

        response = st.write_stream(QueryController.ollama_generator(st.session_state.messages))

    st.session_state.messages[-1]["content"] = question

    st.session_state.messages.append({"role": "assistant", "content": response})
