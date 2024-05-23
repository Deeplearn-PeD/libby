import fire
from libbydbot.brain import embed
import pathlib
import os
import fitz
from fitz import EmptyFileError
from glob import glob
from libbydbot.brain import LibbyDBot


class LibbyInterface:
    def embed(self, corpus_path: str ='.', collection_name: str = 'embeddings'):
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

    def answer(self, question: str, collection_name: str = 'embeddings'):
        DE = embed.DocEmbedder(collection_name)
        context = DE.retrieve_docs(question, num_docs=15)
        LDB = LibbyDBot(model='llama3')
        LDB.set_prompt(f"You are Libby D. Bot, a research Assistant, you should answer questions "
                       f"based on the context provided below.\n{context}")

        response = LDB.ask(question)
        return response



def main(corpus_path='.'):
    fire.Fire(LibbyInterface)



