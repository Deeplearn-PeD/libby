from libbydbot import Persona
import yaml
from base_agent.llminterface import LangModel
from openai import OpenAI
from ollama import Client
import ollama

import dotenv

dotenv.load_dotenv()



class LibbyDBot(Persona):
    def __init__(self, name: str = 'Libby D. Bot', languages=['pt_BR', 'en'], model: str='gpt-4-0125-preview'):
        super().__init__(name=name, languages=languages,model=model)
        self.llm = LangModel(model=model)
        self.prompt_template = None


    def set_prompt(self, prompt_template):
        self.prompt_template = prompt_template

    def ask(self, question: str):
        response = self.get_response(question)
        return response

    def get_response(self, question):
        response =  self.llm.get_response(question=question, context=self.prompt_template)
        return response

    def get_prompt(self):
        return self.context_prompt

