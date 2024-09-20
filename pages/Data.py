
from langchain_community.embeddings.ollama import OllamaEmbeddings
from database_controller import DatabaseController
from langchain_core.documents import Document
from langchain_chroma import Chroma
import streamlit as st

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

DatabaseController = DatabaseController(database, DATA_PATH)

df = DatabaseController.database_to_dataframes()

#=============================================================================#

column_configuration = {
    "source": st.column_config.TextColumn(
        "Source", 
        help="The name of the source", 
        max_chars=100, 
        width="small"
    ),
}

st.set_page_config(layout="wide")

#=============================================================================#

st.header("所有的PDF")

col1, col2 = st.columns([9,1])

df_source = df[['source']].drop_duplicates()

event = col1.dataframe(
    df_source,
    column_config=column_configuration,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
)

if col2.button("更新"):
    DatabaseController.update_db()
    st.rerun()

if col2.button("重置"):
    DatabaseController.reset_db()
    st.rerun()

if col2.button("清除"):
    DatabaseController.clear_db()
    st.rerun()

#-----------------------------------------------------------------------------#

st.header("已選擇的PDF")

col1, col2 = st.columns([9,1])

select_id = event.selection.rows

select_source = df_source.iloc[select_id]

df_selected = df.merge(select_source)

edited_df = col1.data_editor(
    df_selected[["source", "page", "documents"]],
    disabled=["source", "page", "documents"],
    use_container_width=True,
    hide_index=True,
)

if col2.button('刪除選取的PDF'):
    
   delete_ids = df_selected['ids'].values.tolist()

   DatabaseController.clear_database(delete_ids)
   
   st.rerun()
