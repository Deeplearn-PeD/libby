import fire
from libbydbot.brain import embed
import pathlib
import os
import fitz
from fitz import EmptyFileError
from glob import glob
from libbydbot.brain import LibbyDBot
from libbydbot.settings import Settings

try:
    settings = Settings()
except Exception as e:
    print(f"Error loading settings: {e}")
    settings = None

class LibbyInterface(LibbyDBot):
    @staticmethod
    def load_available_models():
        if settings is None:
            return {}
        return {name: details['code'] for name, details in settings.models.items()}

    def __init__(self, name: str='Libby D. Bot', collection_name: str = 'Libby D. Bot', languages=['pt_BR', 'en'], model: str = 'qwen3', dburl: str= 'sqlite:///memory.db',
                 embed_db: str = 'duckdb:///embeddings.duckdb'):
        super().__init__(name=name, languages=languages, model=model, dburl=dburl, embed_db=embed_db)
        if collection_name != name:
            self.DE.collection_name = collection_name

    def embed(self, corpus_path: str ='.'):
        """
        Embed a corpus of documents
        :param corpus_path: path to a folder containing PDFs
        :return:
        """
        print ("Processing your corpus using granular chunking...")
        self.DE.embed_path(corpus_path)
        return self.DE

    def answer(self, question: str, collection_name: str = 'main'):
        """
        Answer a question based on a collection of documents.
        Prioritizes agentic tool-calling if supported by the base agent.
        :param question: Users question
        :param collection_name: collection of documents on which to base the answer
        :return: Answer to the question
        """
        # If the agent has tools registered, we can try to let it handle retrieval
        if hasattr(self.llm, 'agent') and self.llm.agent.tools:
            self.set_prompt(f"You are Libby D. Bot, a research Assistant. Use the search_library tool to find information if needed.")
            # Clear manual context to encourage tool use
            self.set_context("")
            return self.ask(question)
            
        # Fallback to manual RAG
        context = self.DE.retrieve_docs(question, collection=collection_name, num_docs=5)
        self.set_prompt(f"You are Libby D. Bot, a research Assistant")
        self.set_context(context)

        response = self.ask(question)
        return response

    def generate(self, prompt: str='', output_file: str = '', prompt_file: str = ''):
        """
        Generate text based on a prompt
        :param prompt: The prompt to generate text from
        :param output_file: Optional file path to save the generated text
        :param prompt_file: Optional file path to read the prompt from
        :return: Generated text
        """
        if not prompt:
            if prompt_file:
                try:
                    with open(prompt_file, 'r') as f:
                        prompt = f.read().strip()
                except FileNotFoundError:
                    print(f"Error: Prompt file '{prompt_file}' not found.")
                    return
                except Exception as e:
                    print(f"Error reading prompt file: {e}")
                    return
            else:
                prompt = input("Enter a prompt: ")
        # DE = embed.DocEmbedder("embedding")
        context = self.DE.retrieve_docs(prompt,  num_docs=100)
        self.set_prompt("You are Libby D. Bot, a creative and competent writer.")
        self.set_context(context)
        response = self.ask(prompt)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(response)
            print(f"Generated text saved to: {output_file}")
        
        return response


def main(corpus_path='.'):
    fire.Fire(LibbyInterface)



