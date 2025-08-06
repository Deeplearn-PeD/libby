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
    def __init__(self, name: str = 'Libby D. Bot', languages=['pt_BR', 'en'], model: str = 'gpt-4o', dburl: str= 'sqlite:///memory.db'):
        super().__init__(name=name, languages=languages, model=model)
        self.dburl = dburl
        self.llm = LangModel(model=model, provider=PROVIDERS[model])
        self.struct_llm = StructuredLangModel(model=model, provider=PROVIDERS[model])
        self.prompt_template = None
        self.context_prompt = ""
        self.history = History(dburl)

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
