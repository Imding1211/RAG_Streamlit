
from langchain_community.embeddings.ollama import OllamaEmbeddings
from database_controller import DatabaseController
from langchain_chroma import Chroma
import streamlit as st

#=============================================================================#

EMBEDDING_MODEL = "all-minilm"
CHROMA_PATH     = "chroma"

#=============================================================================#

# 初始化Chroma向量存儲
database = Chroma(
    persist_directory  = CHROMA_PATH, 
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    )

DatabaseController = DatabaseController(database)

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
        "總頁數", 
        help="資料總頁數", 
        max_chars=100, 
        width="small"
    ),
    "size": st.column_config.TextColumn(
        "大小", 
        help="資料大小", 
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

st.title("資料庫")

files = st.file_uploader(
    "Upload a PDF file", 
    type="pdf", 
    accept_multiple_files=True, 
    label_visibility="hidden",
    )

col1, col2 = st.columns([9,1])

if col2.button("更新"):
    for file in files:
        DatabaseController.add_PDF_to_chroma(file)

df = DatabaseController.database_to_dataframes()

df_event = df.loc[df.groupby('source')['page'].idxmax(), ['source', 'page', 'size']]

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

st.divider()

st.dataframe(
    df_result[['source', 'page', 'documents']],
    column_config=selected_config,
    use_container_width=True, 
    hide_index=True
    )

if col2.button('刪除'):
    delete_ids = df_result['ids'].values.tolist()
    DatabaseController.clear_database(delete_ids)
    st.rerun()
