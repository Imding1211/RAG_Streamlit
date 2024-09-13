
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import pandas as pd
import uuid

#=============================================================================#

class DatabaseController():

    def __init__(self, database, data_path):
        self.database  = database
        self.data_path = data_path

        self.prompt_templt = """
        你是資料庫的管理員，你收到{mode}資料庫的命令，

        {mode}資料數量為: {doc_num}

        請根據以上資訊簡短使用一句繁體中文回覆。
        """

#-----------------------------------------------------------------------------#

    def populate_database(self):

        documents = self.load_documents()

        chunks = self.split_documents(documents)

        new_chunks = self.add_to_chroma(chunks)

        return new_chunks

#-----------------------------------------------------------------------------#

    def clear_database(self, delete_ids):

        if list(delete_ids):
            self.database.delete(ids=list(delete_ids))

#-----------------------------------------------------------------------------#

    def load_documents(self):

        # 載入指定資料夾中的PDF文件
        document_loader = PyPDFDirectoryLoader(self.data_path)

        return document_loader.load()

#-----------------------------------------------------------------------------#

    def split_documents(self, documents):

        # 使用遞歸字符分割器分割文件
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size         = 800,   # 每塊的大小
            chunk_overlap      = 80,    # 每塊之間的重疊部分
            length_function    = len,   # 用於計算塊長度的函數
            is_separator_regex = False, # 是否使用正則表達式作為分隔符
        )

        return text_splitter.split_documents(documents)

#-----------------------------------------------------------------------------#

    def calculate_chunk_ids(self, chunks):

        # 計算每個塊的唯一ID
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page   = chunk.metadata.get("page")

            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"

            last_page_id = current_page_id

            # 將ID添加到頁面的元數據中
            chunk.metadata["id"] = chunk_id

        return chunks

#-----------------------------------------------------------------------------#

    def calculate_existing_ids(self):

        # 獲取現有文件的ID
        existing_items = self.database.get(include=[])
        existing_ids   = set(existing_items["ids"])

        return existing_ids

#-----------------------------------------------------------------------------#

    def add_to_chroma(self, chunks):

        # 計算每個塊的ID
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # 獲取現有文件的ID
        existing_ids = self.calculate_existing_ids()
        
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            new_chunk_ids = [str(uuid.uuid4()) for _ in range(len(new_chunks))]
            self.database.add_documents(new_chunks, ids=new_chunk_ids)

        return new_chunks

#-----------------------------------------------------------------------------#

    def update_db(self):

        existing_ids = self.calculate_existing_ids()

        new_chunks = self.populate_database()

        prompt = ChatPromptTemplate.from_template(self.prompt_templt)

        prompt = prompt.format(mode="更新", doc_num=len(new_chunks))

        return prompt

#-----------------------------------------------------------------------------#

    def reset_db(self):

        delete_ids = self.calculate_existing_ids()

        self.clear_database(delete_ids)

        existing_ids = self.calculate_existing_ids()

        new_chunks = self.populate_database()

        prompt = ChatPromptTemplate.from_template(self.prompt_templt)

        prompt = prompt.format(mode="重置", doc_num=len(new_chunks))

        return prompt

#-----------------------------------------------------------------------------#

    def clear_db(self):

        delete_ids = self.calculate_existing_ids()

        self.clear_database(delete_ids)

        prompt = ChatPromptTemplate.from_template(self.prompt_templt)

        prompt = prompt.format(mode="清除", doc_num=len(list(delete_ids)))

        return prompt

#-----------------------------------------------------------------------------#

    def database_to_dataframes(self):

        data = self.database.get()

        df = pd.DataFrame(columns=['ids', 'documents', 'id', 'page', 'source'])

        for index, (ids, documents, metadatas) in enumerate(zip(data["ids"], data["documents"], data["metadatas"])):
            df.loc[index] = [ids, documents, metadatas['id'], metadatas['page'], metadatas['source'].split('/')[-1]]

        return df

#-----------------------------------------------------------------------------#

    def update_documents(self, update_ids, update_documents):

        self.database.update_documents(ids=update_ids, documents=update_documents)


