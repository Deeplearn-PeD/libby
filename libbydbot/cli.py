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

    def __init__(self, name: str='Libby D. Bot', collection_name: str = 'Libby D. Bot', languages=['pt_BR', 'en'], model: str = None, dburl: str= 'sqlite:///memory.db',
                 embed_db: str = 'duckdb:///embeddings.duckdb'):
        available_models = self.load_available_models()
        self.collection_name = collection_name
        if model is None:
            model = settings.default_model
        elif model in available_models:
            model = available_models[model]
        else:
            raise ValueError(f"Invalid model. Available models: {', '.join(available_models.keys())}")
            
        super().__init__(name=name, languages=languages, model=model, dburl=dburl)
        self.DE = embed.DocEmbedder(col_name=collection_name, dburl=embed_db)

    def embed(self, corpus_path: str ='.'):
        """
        Embed a corpus of documents
        :param corpus_path: path to a folder containing PDFs
        :param collection_name: Name of the document collection
        :return:
        """
        # DE = embed.DocEmbedder(collection_name, dburl=dburl)
        print ("Processing your corpus...")
        for d in glob(os.path.join(corpus_path, '*.pdf')):
            try:
                doc = fitz.open(d)
            except EmptyFileError:
                continue
            n = doc.name
            for page_number, page in enumerate(doc):
                text = page.get_text()
                if not text:
                    continue
                self.DE.embed_text(text, n, page_number)
        return self.DE

    def answer(self, question: str, collection_name: str = 'main'):
        """
        Answer a question based on a collection of documents
        :param question: Users question
        :param collection_name: collection of documents on which to base the answer
        :return: Answer to the question
        """
        # DE = embed.DocEmbedder(collection_name)
        context = self.DE.retrieve_docs(question, collection=collection_name, num_docs=5)
        # LDB = LibbyDBot(model='llama3')
        self.set_prompt(f"You are Libby D. Bot, a research Assistant")
        self.set_context(context)

        response = self.ask(question)
        return response

    def generate(self, prompt: str=None, output_file: str = None, prompt_file: str = None):
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



