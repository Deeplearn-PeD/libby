import fire
from libbydbot.brain import embed
import pathlib
import os
import fitz
from glob import glob


class LibbyInterface:
    def embed_PDFs(self, corpus_path: str ='.', collection_name: str = 'default'):
        DE = embed.DocEmbedder()
        print ("Processing your corpus...")
        for d in glob(os.path.join(corpus_path, '*.pdf')):
            doc = fitz.open(d)
            n = doc.name
            for page_number, page in enumerate(doc):
                DE.embed_text(page.get_text(), n, page_number)
        return DE

    def answer(self, question: str, collection_name: str = 'default'):
        DE = embed.DocEmbedder()
        if collection_name in DE.embeddings:
            DE.schema = DE.embeddings[collection_name]
            response = DE.generate_response(question)

        return response



def main(corpus_path='.'):
    fire.Fire(LibbyInterface)



