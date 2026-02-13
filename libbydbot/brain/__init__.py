from libbydbot import Persona
from base_agent.llminterface import LangModel, StructuredLangModel
from .memory import History
import loguru
logger = loguru.logger

PROVIDERS = {
    "llama3.2": "llama",
    "gemma3": "google",
    "gpt-4o": "openai",
    "qwen3": "qwen",
    "gemini": "google"
}

class LibbyDBot(Persona):
    def __init__(self, name: str = 'Libby D. Bot', languages=['pt_BR', 'en'], model: str = 'gpt-4o', dburl: str= 'sqlite:///memory.db', provider: str='google', embed_db: str = 'duckdb:///embeddings.duckdb'):
        super().__init__(name=name, languages=languages, model=model)
        self.dburl = dburl
        self.llm = LangModel(model=model, provider=PROVIDERS.get(model, 'google'))
        self.struct_llm = StructuredLangModel(model=model, )
        self.prompt_template = None
        self.context_prompt = ""
        self.history = History(dburl)
        
        # doc_embedder for RAG
        from .embed import DocEmbedder
        self.DE = DocEmbedder(col_name=name, dburl=embed_db)
        
        # Register retrieval as a tool if the agent supports it
        if hasattr(self.llm, 'agent'):
            @self.llm.agent.tool_plain
            def search_library(query: str) -> str:
                """Search the document library for relevant information using hybrid search.
                
                Args:
                    query: The search query.
                """
                logger.info(f"Agent is searching library for: {query}")
                return self.DE.retrieve_docs(query)

    @property
    def context(self):
        return self.context_prompt

    def set_context(self, context):
        self.context_prompt = context

    def set_prompt(self, prompt_template):
        self.prompt_template = prompt_template

    def ask(self, question: str, user_id: int=1):
        response = self._get_response(question)
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        self.history.memorize(user_id, question, response, self.context)
        return response

    # def memorize(self, question: str, response: str):
    #     self.set_context(question)
    #     self.set_prompt(f"You are Libby D. Bot, a research Assistant, you should answer questions "
    #                    f"based on the context provided below.\n{question}")
    #     self.ask(response)
    #     return True

    def _get_response(self, question):
        response = self.llm.get_response(question=question, context=self.context_prompt)
        return response
