from libbydbot import Persona
from base_agent.llminterface import LangModel, StructuredLangModel
from .memory import memorize, remember



class LibbyDBot(Persona):
    def __init__(self, name: str = 'Libby D. Bot', languages=['pt_BR', 'en'], model: str = 'gpt-4o'):
        super().__init__(name=name, languages=languages, model=model)
        self.llm = LangModel(model=model)
        self.struct_llm = StructuredLangModel(model=model)
        self.prompt_template = None
        self.context_prompt = ""

    @property
    def context(self):
        return self.context_prompt

    def set_context(self, context):
        self.context_prompt = context

    def set_prompt(self, prompt_template):
        self.prompt_template = prompt_template

    def ask(self, question: str):
        response = self._get_response(question)
        memorize(1, question, response, self.context)
        return response

    def memorize(self, question: str, response: str):
        self.set_context(question)
        self.set_prompt(f"You are Libby D. Bot, a research Assistant, you should answer questions "
                       f"based on the context provided below.\n{question}")
        self.ask(response)
        return True

    def _get_response(self, question):
        response = self.llm.get_response(question=question, context=self.context_prompt)
        return response
