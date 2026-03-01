from libbydbot import Persona
from libbydbot.settings import Settings
from base_agent.llminterface import LangModel, StructuredLangModel
from .memory import History
import loguru

logger = loguru.logger

settings = Settings()


# Monkeypatch base_agent.llminterface.LangModel.get_response
# to fix Agent.run() argument mismatch (system_prompt is not an argument in pydantic_ai Agent.run)
def patched_get_response(self, question: str, context: str = ""):
    if not self.agent:
        self._setup_llm_client(self.provider or "ollama")

    import asyncio
    import nest_asyncio

    nest_asyncio.apply()

    async def _run():
        history = self.chat_history.get_all()
        # combine context into question for robustness if system_prompt fails
        full_prompt = (
            f"Context: {context}\n\nQuestion: {question}" if context else question
        )
        result = await self.agent.run(full_prompt, message_history=history)
        for msg in result.new_messages():
            self.chat_history.enqueue(msg)
        return result.data

    return asyncio.run(_run())


def patched_structured_get_response(
    self, question: str, context: str = "", response_model=None
):
    if not self.agent:
        self._setup_llm_client(self.provider or "ollama")

    import asyncio
    import nest_asyncio

    nest_asyncio.apply()

    async def _run():
        history = self.chat_history.get_all()
        full_prompt = (
            f"Context: {context}\n\nQuestion: {question}" if context else question
        )
        result = await self.agent.run(
            full_prompt, message_history=history, result_type=response_model
        )
        for msg in result.new_messages():
            self.chat_history.enqueue(msg)
        return result.data

    return asyncio.run(_run())


LangModel.get_response = patched_get_response
StructuredLangModel.get_response = patched_structured_get_response

PROVIDERS = {
    "llama3.2": "llama",
    "gemma3": "google",
    "gemma": "google",
    "gpt-4o": "openai",
    "qwen3": "qwen",
    "gemini": "google",
}


class LibbyDBot(Persona):
    def __init__(
        self,
        name: str = "Libby D. Bot",
        languages=["pt_BR", "en"],
        model: str = "llama3.2",
        dburl: str = "sqlite:///memory.db",
        provider: str = None,
        embed_db: str = "duckdb:///embeddings.duckdb",
    ):
        if model.lower() == "gemma":
            model = "gemma3"

        if model not in PROVIDERS:
            raise ValueError(
                f"Model {model} not supported. Supported models: {list(PROVIDERS.keys())}"
            )

        super().__init__(name=name, languages=languages, model=model)
        self.dburl = dburl
        self.llm = LangModel(model=model, provider=PROVIDERS.get(model, "google"))
        self.struct_llm = StructuredLangModel(
            model=model,
        )
        self.prompt_template = None
        self.context_prompt = ""
        self.history = History(dburl)

        # doc_embedder for RAG
        from .embed import DocEmbedder

        self.DE = DocEmbedder(
            col_name=name,
            dburl=embed_db,
            embedding_model=settings.default_embedding_model,
        )

        # Register retrieval as a tool if the agent supports it
        if hasattr(self.llm, "agent"):

            @self.llm.agent.tool_plain
            def search_library(query: str) -> str:
                """Search the document library for relevant information using hybrid search.

                Args:
                    query: The search query.
                """
                logger.info(f"Agent is searching library for: {query}")
                return self.DE.retrieve_docs(query)

    @property
    def context(self):
        return self.context_prompt

    def set_context(self, context):
        self.context_prompt = context

    def set_prompt(self, prompt_template):
        self.prompt_template = prompt_template

    def ask(self, question: str, user_id: int = 1):
        response = self._get_response(question)
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        self.history.memorize(user_id, question, response, self.context)
        return response

    # def memorize(self, question: str, response: str):
    #     self.set_context(question)
    #     self.set_prompt(f"You are Libby D. Bot, a research Assistant, you should answer questions "
    #                    f"based on the context provided below.\n{question}")
    #     self.ask(response)
    #     return True

    def _get_response(self, question):
        response = self.llm.get_response(question=question, context=self.context_prompt)
        return response
