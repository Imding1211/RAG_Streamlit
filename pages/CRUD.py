
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
import streamlit as st
import pandas as pd

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

st.set_page_config(layout="wide")

#=============================================================================#

def database_to_dataframes(database):

    data = database.get()

    df = pd.DataFrame(columns=['ids', 'documents', 'page', 'source'])

    for index, (ids, documents, metadatas) in enumerate(zip(data["ids"], data["documents"], data["metadatas"])):
        df.loc[index] = [ids, documents, metadatas['page'], metadatas['source']]

    return df

#=============================================================================#

df = database_to_dataframes(DATABASE)

st.dataframe(df)

