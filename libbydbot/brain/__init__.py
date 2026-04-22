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

    async def _run():
        history = self.chat_history.get_all()
        full_prompt = (
            f"Context: {context}\n\nQuestion: {question}" if context else question
        )
        result = await self.agent.run(full_prompt, message_history=history)
        for msg in result.new_messages():
            self.chat_history.enqueue(msg)
        response = result.response
        # Extract text from ModelResponse if needed (pydantic-ai v2 API)
        from pydantic_ai.messages import ModelResponse as _MR
        if isinstance(response, _MR):
            text_parts = [
                p.content for p in response.parts
                if hasattr(p, "content") and isinstance(p.content, str)
            ]
            return "\n".join(text_parts) if text_parts else str(response)
        return response

    nest_asyncio.apply()
    try:
        return asyncio.run(_run())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # Fallback: get the running loop and run until complete
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run())
        raise


def patched_structured_get_response(
    self, question: str, context: str = "", response_model=None
):
    if not self.agent:
        self._setup_llm_client(self.provider or "ollama")

    import asyncio
    import nest_asyncio

    async def _run():
        history = self.chat_history.get_all()
        full_prompt = (
            f"Context: {context}\n\nQuestion: {question}" if context else question
        )
        kwargs = {"message_history": history}
        if response_model is not None:
            kwargs["output_type"] = response_model
        result = await self.agent.run(full_prompt, **kwargs)
        for msg in result.new_messages():
            self.chat_history.enqueue(msg)
        response = result.response
        from pydantic_ai.messages import ModelResponse as _MR
        if isinstance(response, _MR):
            text_parts = [
                p.content for p in response.parts
                if hasattr(p, "content") and isinstance(p.content, str)
            ]
            return "\n".join(text_parts) if text_parts else str(response)
        return response

    nest_asyncio.apply()
    try:
        return asyncio.run(_run())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run())
        raise


LangModel.get_response = patched_get_response
StructuredLangModel.get_response = patched_structured_get_response


# Monkeypatch base_agent.llminterface.LangModel._get_model_instance
# to fix OpenAIChatModel initialization with newer pydantic-ai API.
# The old API passed api_key/base_url directly; new API uses OpenAIProvider.
def patched_get_model_instance(self, provider: str, model_name: str):
    """Create a Pydantic AI Model instance based on provider and config."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider_config = getattr(self, "_config", None) or {}
    # Fallback: base_agent stores config in CONFIG module variable
    import base_agent.llminterface as _bali

    provider_config = (_bali.CONFIG or {}).get("providers", {}).get(provider, {})
    base_url = provider_config.get("base_url")
    api_key = self.keys.get(provider)

    if provider == "ollama" and base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url.rstrip('/')}/v1"

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(api_key=api_key or "dummy", base_url=base_url),
    )


LangModel._get_model_instance = patched_get_model_instance


# Monkeypatch base_agent.llminterface.LangModel._set_active_model
# to avoid fallback to wrong model when model name contains a tag suffix (e.g. qwen3.5:4b).
def patched_set_active_model(self, model: str):
    found_provider = self._find_model_provider(model)
    if found_provider:
        self.model = model
        self._setup_llm_client(found_provider)
    else:
        # Try matching without the tag suffix (e.g. "qwen3.5:4b" → "qwen3.5")
        base_model = model.split(":")[0]
        for provider, models_list in self.provider_models.items():
            for m in models_list:
                if m == base_model or m.startswith(base_model):
                    self.model = model
                    self._setup_llm_client(provider)
                    return
        # Last resort: try ollama directly if it's a known local model
        import os

        try:
            import httpx

            host = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            resp = httpx.get(f"{host}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    if m["name"] == model or m["name"].startswith(model.split(":")[0]):
                        self.model = model
                        self._setup_llm_client("ollama")
                        return
        except Exception:
            pass

        # Original fallback
        fallback_model = (
            "qwen3" if "qwen3" in self.available_models else
            (self.available_models[-1] if self.available_models else model)
        )
        print(f"Model {model} not found. Using fallback: {fallback_model}")
        self.model = fallback_model
        provider = self._find_model_provider(self.model) or "ollama"
        self._setup_llm_client(provider)


LangModel._set_active_model = patched_set_active_model

PROVIDERS = {
    "llama3.2": "llama",
    "gemma3": "google",
    "gemma": "google",
    "gpt-4o": "openai",
    "qwen3": "qwen",
    "qwen3.5:4b": "qwen",
    "gemini": "google",
    "kimi-k2.5": "moonshot",
}


class LibbyDBot(Persona):
    def __init__(
        self,
        name: str = "Libby D. Bot",
        languages=["pt_BR", "en"],
        model: str = "llama3.2",
        dburl: str = "postgresql://libby:libby123@localhost:5432/libby",
        provider: str = None,
        embed_db: str = "postgresql://libby:libby123@localhost:5432/libby",
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

    def _get_response(self, question):
        response = self.llm.get_response(question=question, context=self.context_prompt)
        return response
