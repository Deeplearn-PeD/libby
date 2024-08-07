import fire
from libbydbot.brain import embed
import pathlib
import os
import fitz
from fitz import EmptyFileError
from glob import glob
from libbydbot.brain import LibbyDBot


class LibbyInterface(LibbyDBot):

    def __init__(self, name: str = 'Libby D. Bot', languages=['pt_BR', 'en'], model: str = 'gpt-4o', dburl: str= 'sqlite:///memory.db'):
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
        self.set_prompt(f"You are Libby D. Bot, a research Assistant, you should answer questions "
                       f"based on the context provided below.\n\n{context}")

        response = self.ask(question)
        return response



def main(corpus_path='.'):
    fire.Fire(LibbyInterface)



