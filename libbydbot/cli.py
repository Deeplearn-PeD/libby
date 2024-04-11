import fire
from libbydbot.brain import embed
import pathlib
import os
import fitz
from glob import glob


def main(corpus_path='.'):
    DE = embed.DocEmbedder()
    print ("Processing your corpus...")
    for d in glob(os.path.join(corpus_path, '*.pdf')):
        doc = fitz.open(d)
        n = doc.name
        for page_number, page in enumerate(doc):
            DE.embed_text(page.get_text(), n, page_number)
    question = input("Ask me a question: ")
    response = DE.generate_response(question)
    print(response)

if __name__ == '__main__':
    fire.Fire(main)

