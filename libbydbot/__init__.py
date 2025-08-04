"""
This package defines a basic AI bot's Personality.
Starting with a Base class setting the basic parameters
for Such as name, language model uses, and basic context sqlprompts defining its purpose.
"""
from base_agent.voice import talk
from libbydbot.persona_prompts import base_prompt
from libbydbot.settings import Settings
from typing import List, Dict, Any, Union
from base_agent import BasePersona
import yaml
import os

import dotenv

dotenv.load_dotenv()

try:
    settings = Settings()
except Exception as e:
    print(f"Error loading settings: {e}")
    settings = None

class Persona(BasePersona):
    def __init__(self, name: str='Libby D. Bot', model: str='gpt-4o',  languages: List=['pt_BR','en'], ):
        super().__init__(name=name, model=model, languages=languages)
        self.name = name
        self.languages = languages
        self.active_language = languages[0]
        self.set_language(self.active_language)
        self.context_prompt = base_prompt[self.active_language]

    def set_language(self, language: str):
        if language in self.languages:
            self.active_language = language
            self.voice = talk.Speaker(language=self.active_language)
            self.say = self.voice.say
            self.context_prompt = base_prompt[self.active_language]
        else:
            raise ValueError(f"Language {language} not supported by this persona.")

