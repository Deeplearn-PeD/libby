from libbydbot import Persona
import yaml
from openai import OpenAI
from ollama import Client
import ollama

import dotenv

dotenv.load_dotenv()


class LLLModel:
    def __init__(self, model: str = 'gpt-4-0125-preview'):
        self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) if 'gpt' in model else Client(host='http://localhost:11434')
        self.model = model

    def get_response(self, question: str, context: str = None) -> str:
        if 'gpt' in self.model:
            return self.get_gpt_response(question, context)
        elif 'gemma' in self.model:
            return self.get_gemma_response(question, context)

    def get_gpt_response(self, question: str, context: str)->str:
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'system',
                    'content': context
                },
                {
                    'role': 'user',
                    'content': question
                }
            ],
            max_tokens=100,
            temperature=0.5,
            top_p=1
        )
        return response.choices[0].message.content

    def get_gemma_response(self, question: str, context: str) -> str:
        response = ollama.generate(
            model=self.model,
            system=context[:2048],
            prompt=question,
        )

        return response['response']
        # return '/n'.join([resp['response'] for resp in response ])
class RegDBot(Persona):
    def __init__(self, name: str = 'Libby D. Bot', languages=['pt_BR', 'en'], model: str='gpt-4-0125-preview'):
        super().__init__(name=name, languages=languages,model=model)
        self.llm = LLLModel(model=model)
        self.prompt_template = None


    def set_prompt(self, prompt_template):
        self.prompt_template = prompt_template

    def ask(self, question: str):
        response = self.get_response(question)
        return response

    def get_response(self, question):
        response =  self.llm.get_response(question=question, context=self.prompt_template.get_prompt())
        return response

    def get_prompt(self):
        return self.context_prompt

