
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Generator
import ollama

#=============================================================================#

class QueryController():

    def __init__(self, database, llm_model, query_num):
        self.database  = database
        self.llm_model = llm_model
        self.query_num = query_num

        self.prompt_templt = """

        {context}

        ---

        根據以上資料用繁體中文回答問題: {question}
        """
        
#-----------------------------------------------------------------------------#

    def generate_results(self, query_text):
        
        # 進行相似度搜索
        query_results = self.database.similarity_search_with_score(query_text, k=self.query_num)

        return query_results

#-----------------------------------------------------------------------------#

    def generate_prompt(self, query_text, query_results):
        
        # 構建上下文文本
        context_text    = "\n\n---\n\n".join([doc.page_content for doc, _score in query_results])
        prompt_template = ChatPromptTemplate.from_template(self.prompt_templt)
        prompt          = prompt_template.format(context=context_text, question=query_text)

        return prompt

#-----------------------------------------------------------------------------#

    def ollama_generator(self, messages: Dict) -> Generator:
        
        stream = ollama.chat(model=self.llm_model, messages=messages, stream=True)
        
        for chunk in stream:
            yield chunk['message']['content']



