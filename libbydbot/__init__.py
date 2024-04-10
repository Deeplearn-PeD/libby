"""
This package defines a basic AI bot's Personality.
Starting with a Base class setting the basic parameters
for Such as name, language model uses, and basic context sqlprompts defining its purpose.
"""
from libbydbot.voice import talk
from libbydbot.persona_prompts import base_prompt
from typing import List, Dict, Any, Union



class Persona:
    def __init__(self, name: str='Libby D. Bot', model: str='gpt-4-0125-preview',  languages=['pt_BR','en'], ):
        self.name = name
        self.languages = languages
        self.active_language = languages[0]
        self.model = model
        self.voice = talk.Speaker(language=self.active_language)
        self.say = self.voice.say
        self.context_prompt = base_prompt[self.active_language]

    def set_language(self, language: str):
        if language in self.languages:
            self.active_language = language
            self.voice = talk.Speaker(language=self.active_language)
            self.say = self.voice.say
            self.context_prompt = base_prompt[self.active_language]
        else:
            raise ValueError(f"Language {language} not supported by this persona.")

