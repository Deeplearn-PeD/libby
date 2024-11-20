import fire
from libbydbot.brain import embed
import pathlib
import os
import fitz
import yaml
from fitz import EmptyFileError
from glob import glob
from libbydbot.brain import LibbyDBot


class LibbyInterface(LibbyDBot):
    
    @staticmethod
    def load_available_models():
        config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return {name: details['code'] for name, details in config['models'].items()}

    def __init__(self, name: str = 'Libby D. Bot', languages=['pt_BR', 'en'], model: str = None, dburl: str= 'sqlite:///memory.db'):
        available_models = self.load_available_models()
        if model is None:
            # Find default model from config
            for model_name, details in yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yml')))['models'].items():
                if details.get('is_default'):
                    model = details['code']
                    break
        elif model in available_models:
            model = available_models[model]
        else:
            raise ValueError(f"Invalid model. Available models: {', '.join(available_models.keys())}")
            
        super().__init__(name=name, languages=languages, model=model, dburl=dburl)

    def embed(self, corpus_path: str ='.', collection_name: str = 'embeddings'):
        """
        Embed a corpus of documents
        :param corpus_path: path to a folder containing PDFs
        :param collection_name: Name of the document collection
        :return:
        """
        DE = embed.DocEmbedder(collection_name)
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
                DE.embed_text(text, n, page_number)
        return DE

    def answer(self, question: str, collection_name: str = 'main'):
        """
        Answer a question based on a collection of documents
        :param question: Users question
        :param collection_name: collection of documents on which to base the answer
        :return: Answer to the question
        """
        DE = embed.DocEmbedder(collection_name)
        context = DE.retrieve_docs(question, collection=collection_name, num_docs=5)
        # LDB = LibbyDBot(model='llama3')
        self.set_prompt(f"You are Libby D. Bot, a research Assistant")
        self.set_context(context)

        response = self.ask(question)
        return response

    def generate(self, prompt: str, output_file: str = None):
        """
        Generate text based on a prompt
        :param prompt: The prompt to generate text from
        :param output_file: Optional file path to save the generated text
        :return: Generated text
        """
        DE = embed.DocEmbedder("embedding")
        context = DE.retrieve_docs(prompt,  num_docs=100)
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



