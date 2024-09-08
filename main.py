
from database_controller import populate_database, clear_database, calculate_existing_ids
from langchain_community.embeddings.ollama import OllamaEmbeddings
from query_controller import generate_results, generate_prompt
from langchain_core.prompts import ChatPromptTemplate
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

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "使用繁體中文回答問題"}]

#=============================================================================#

QUERY_PROMPT_TEMPLATE = """

{context}

---

根據以上資料用繁體中文回答問題: {question}
"""

CMD_PROMPT_TEMPLATE = """

你是管理資料庫得AI，你收到{mode}資料庫的命令，

{mode}資料數量為: {doc_num}

請根據以上資訊簡短使用一句繁體中文回覆，
"""

#=============================================================================#

def ollama_generator(model_name: str, messages: Dict) -> Generator:
    
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    
    for chunk in stream:
        yield chunk['message']['content']

#-----------------------------------------------------------------------------#

def update_db():

    existing_ids = calculate_existing_ids(DATABASE)

    new_chunks = populate_database(EMBEDDING_MODEL, DATA_PATH, DATABASE)

    prompt = ChatPromptTemplate.from_template(CMD_PROMPT_TEMPLATE)
    prompt = prompt.format(mode="更新", doc_num=len(new_chunks))

    return prompt

#-----------------------------------------------------------------------------#

def reset_db():

    delete_ids = calculate_existing_ids(DATABASE)
    clear_database(delete_ids, DATABASE)

    existing_ids = calculate_existing_ids(DATABASE)
    new_chunks = populate_database(EMBEDDING_MODEL, DATA_PATH, DATABASE)

    prompt = ChatPromptTemplate.from_template(CMD_PROMPT_TEMPLATE)
    prompt = prompt.format(mode="重設", doc_num=len(new_chunks))

    return prompt

#-----------------------------------------------------------------------------#

def clear_db():

    delete_ids = calculate_existing_ids(DATABASE)
    clear_database(delete_ids, DATABASE)

    prompt = ChatPromptTemplate.from_template(CMD_PROMPT_TEMPLATE)
    prompt = prompt.format(mode="清除", doc_num=len(list(delete_ids)))

    return prompt

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
        prompt = update_db()

    elif "重設資料庫" in question:
        prompt = reset_db()

    elif "清除資料庫" in question:
        prompt = clear_db()

    else:
        results = generate_results(question, QUERY_NUM, DATABASE)
        prompt  = generate_prompt(question, results, QUERY_PROMPT_TEMPLATE)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):

        response = st.write_stream(ollama_generator(LLM_MODEL_NAME, st.session_state.messages))

    st.session_state.messages[-1]["content"] = question

    st.session_state.messages.append({"role": "assistant", "content": response})
