
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
    "page": st.column_config.TextColumn(
        "Page", 
        help="The page of the source", 
        max_chars=100, 
        width="small"
    ),
    "documents": st.column_config.TextColumn(
        "Content", 
        help="The content of the source",  
        width="medium"
    ),
}

st.set_page_config(layout="wide")

#=============================================================================#

st.header("All Data")

col1, col2 = st.columns([9,1])

event = col1.dataframe(
    df[['source', 'page', 'documents']],
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

st.header("Selected Data")

col1, col2 = st.columns([9,1])

select_id = event.selection.rows

edited_df = col1.data_editor(
    df.iloc[select_id],
    disabled=["ids", "source", "page"],
    use_container_width=True,
    hide_index=True,
)

if col2.button('刪除選取資料'):
    
   delete_ids = df.loc[select_id, ['ids']]
   delete_ids = delete_ids['ids'].values.tolist()

   DatabaseController.clear_database(delete_ids)
   
   st.rerun()

if col2.button('更新選取資料'):

   update_documents = []
   update_ids       = []

   for index, row in edited_df.iterrows():
      update_document = Document(
        page_content=row['documents'], 
        metadata={
        'id': row['id'], 
        'page': row['page'],
        'source': row['source']
        })

      update_ids.append(row['ids'])
      update_documents.append(update_document)

   DatabaseController.update_documents(update_ids, update_documents)

   st.rerun()
