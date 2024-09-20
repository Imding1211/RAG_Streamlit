
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import streamlit as st
import pandas as pd
import PyPDF2
import uuid

#=============================================================================#

EMBEDDING_MODEL = "all-minilm"
CHROMA_PATH     = "chroma"
DATA_PATH       = "data"

#=============================================================================#

# 初始化Chroma向量存儲
database = Chroma(
    persist_directory  = CHROMA_PATH, 
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size         = 800,   # 每塊的大小
    chunk_overlap      = 80,    # 每塊之間的重疊部分
    length_function    = len,   # 用於計算塊長度的函數
    is_separator_regex = False, # 是否使用正則表達式作為分隔符
)

#=============================================================================#

st.set_page_config(layout="wide")

event_config = {
    "source": st.column_config.TextColumn(
        "資料", 
        help="資料名稱", 
        max_chars=100, 
        width="small"
    ),
    "page": st.column_config.TextColumn(
        "頁數", 
        help="資料頁數", 
        max_chars=100, 
        width="small"
    ),
}

selected_config = {
    "source": st.column_config.TextColumn(
        "資料", 
        help="資料名稱", 
        max_chars=100, 
        width="small"
    ),
    "page": st.column_config.TextColumn(
        "頁數", 
        help="資料頁數", 
        max_chars=100, 
        width="small"
    ),
    "documents": st.column_config.TextColumn(
        "內容", 
        help="資料內容", 
        max_chars=100, 
        width="small"
    ),
}

#=============================================================================#

st.header("PDF")

files = st.file_uploader(
    "Upload a PDF file", 
    type="pdf", 
    accept_multiple_files=True, 
    label_visibility="hidden"
    )

col1, col2 = st.columns([9,1])

if col2.button("更新"):

    for file in files:

        pdf_reader = PyPDF2.PdfReader(file)

        for page in range(len(pdf_reader.pages)):

            content = pdf_reader.pages[page].extract_text()

            documents = text_splitter.create_documents([content], [{"source":file.name, "page":page}])

            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

            database.add_documents(documents, ids=ids)

data = database.get()

df = pd.DataFrame(columns=['ids', 'documents', 'page', 'source'])

for index, (ids, documents, metadatas) in enumerate(zip(data["ids"], data["documents"], data["metadatas"])):
    df.loc[index] = [ids, documents, metadatas['page'], metadatas['source']]

df_event = df.loc[df.groupby('source')['page'].idxmax(), ['source', 'page']]

event = col1.dataframe(
    df_event,
    column_config=event_config,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    )

select_id = event.selection.rows

df_selected = df_event.iloc[select_id][['source']]

df_result = df.merge(df_selected, on='source')

st.dataframe(
    df_result[['source', 'page', 'documents']],
    column_config=selected_config,
    use_container_width=True, 
    hide_index=True
    )

if col2.button('刪除'):
    delete_ids = df_result['ids'].values.tolist()
    if delete_ids:
        database.delete(ids=delete_ids)
        st.rerun()