import fire
from libbydbot.brain import embed
import pathlib
import os
import fitz
from fitz import EmptyFileError
from glob import glob


class LibbyInterface:
    def embed_PDFs(self, corpus_path: str ='.', collection_name: str = 'embeddings'):
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
        # if collection_name in DE.embeddings_list:
        #     DE.set_schema(collection_name)
        response = DE.generate_response(question)

        return response



def main(corpus_path='.'):
    fire.Fire(LibbyInterface)



