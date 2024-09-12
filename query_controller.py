
from langchain_core.prompts import ChatPromptTemplate

#=============================================================================#

class QueryController():

    def __init__(self, database, llm_model, query_num):
        self.database  = database
        self.llm_model = llm_model
        self.query_num = query_num

#-----------------------------------------------------------------------------#

    def query_rag(self, query_text, prompt_template):
        
        results  = self.generate_results(query_text)
        
        prompt   = self.generate_prompt(query_text, results, prompt_template)
        
        response = self.generate_response(prompt, results)
        
        return response

#-----------------------------------------------------------------------------#

    def generate_results(self, query_text):
        
        # 進行相似度搜索
        query_results = self.database.similarity_search_with_score(query_text, k=self.query_num)

        return query_results

#-----------------------------------------------------------------------------#

    def generate_prompt(self, query_text, query_results, prompt_template):
        
        # 構建上下文文本
        context_text    = "\n\n---\n\n".join([doc.page_content for doc, _score in query_results])
        prompt_template = ChatPromptTemplate.from_template(prompt_template)
        prompt          = prompt_template.format(context=context_text, question=query_text)

        return prompt

#-----------------------------------------------------------------------------#

    def generate_response(self, prompt, query_results):

        # 生成回覆
        response_text = llm_model.invoke(prompt)

        # 格式化並輸出回應
        sources  = [doc.metadata.get("id", None) for doc, _score in query_results]
        response = f"Response: {response_text}"

        return response



