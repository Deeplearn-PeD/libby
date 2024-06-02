'''
In this module, we define the schema for the article summary table
'''

from typing import List, Tuple
from pydantic import BaseModel, Field
from base_agent.llminterface import StructuredLangModel
import fitz

class ArticleSummary(BaseModel):
    title: str = Field(..., title="Title of the article")
    summary: str = Field(..., title="Summary of the article")
    research_question: str = Field(..., title="Research question of the article")
    keywords: List[str] = Field(..., title="Keywords of the article")
    results: List[str] = Field(..., title="Results of the article")
    conclusions: List[str] = Field(..., title="Conclusions of the article")


class ArticleSummarizer:
    def __init__(self, model='llama3'):
        self.model = StructuredLangModel(model=model)
        self.prompt = "You a researcher studying the scientific literature. You should summarize scientific articles, extracting aspects such as its main research question, results and conclusions.\n\n"

    def summarize(self, article: str) -> ArticleSummary:
        resp = self.model.get_response('please summarize the article below:\n\n' + article, self.prompt, ArticleSummary)
        return resp

if __name__ == '__main__':
    import pymupdf4llm
    pth = "/home/fccoelho/Documentos/pdfs/Wei et al_2008_An epidemic model of a vector-borne disease with direct transmission and time.pdf"

    article = pymupdf4llm.to_markdown(pth)
    # print(article)
    AS = ArticleSummarizer()
    summary = AS.summarize(article)
    print(summary)

