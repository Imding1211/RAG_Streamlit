
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import PyPDF2
import uuid

#=============================================================================#

class DatabaseController():

    def __init__(self, database):
        self.database  = database

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size         = 800,   # 每塊的大小
            chunk_overlap      = 80,    # 每塊之間的重疊部分
            length_function    = len,   # 用於計算塊長度的函數
            is_separator_regex = False, # 是否使用正則表達式作為分隔符
            )

#-----------------------------------------------------------------------------#

    def calculate_existing_ids(self):

        existing_items = self.database.get(include=[])
        existing_ids   = set(existing_items["ids"])

        return existing_ids

#-----------------------------------------------------------------------------#

    def database_to_dataframes(self):

        data = self.database.get()

        df = pd.DataFrame({
            'ids'       : data['ids'],
            'page'      : [meta['page'] for meta in data['metadatas']],
            'source'    : [meta['source'] for meta in data['metadatas']],
            'size'      : [meta['size'] for meta in data['metadatas']],
            'documents' : data['documents']
        })

        return df

#-----------------------------------------------------------------------------#

    def clear_database(self, delete_ids):
        if delete_ids:
            self.database.delete(ids=delete_ids)

#-----------------------------------------------------------------------------#

    def add_PDF_to_chroma(self, file):

        pdf = PyPDF2.PdfReader(file)

        for page in range(len(pdf.pages)):
            
            content = pdf.pages[page].extract_text()

            metadata = {"source":pdf.stream.name, "page":page, "size":pdf.stream.size}

            documents = self.text_splitter.create_documents([content], [metadata])

            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

            if len(documents):
                self.database.add_documents(documents, ids=ids)
