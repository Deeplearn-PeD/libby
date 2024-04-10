import fire
from libbydbot.brain import embed
import pathlib
import os
import fitz
from glob import glob


def main(corpus_path='.'):
    print ("Processing your corpus...")
    for d in glob(os.path.join(corpus_path, '*.pdf')):
        doc = fitz.open(d)
        for page in doc:
            embed.embed_docs(page.get_text())
    question = input("Ask me a question: ")
    response = embed.generate_response(question)
    print(response)

if __name__ == '__main__':
    fire.Fire(main)

