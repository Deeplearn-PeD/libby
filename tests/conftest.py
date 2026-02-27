import pytest
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture(autouse=True)
def mock_embeddings():
    # Mocking Ollama/Gemini embeddings globally
    with patch("libbydbot.brain.embed.DocEmbedder._generate_embedding") as mocked:
        mocked.return_value = np.zeros(1024).tolist()
        yield mocked


@pytest.fixture(autouse=True, scope="session")
def check_postgres():
    # This can be used to skip postgres tests if env is not ready
    import os

    if not os.getenv("PGURL"):
        # We don't skip the whole session, just individual tests usually
        pass


@pytest.fixture(autouse=True, scope="session")
def mock_llm():
    """
    Mock LangModel and StructuredLangModel to prevent real API calls during tests.
    """
    from base_agent.llminterface import LangModel, StructuredLangModel
    import asyncio
    import nest_asyncio

    # Mock the _setup_llm_client to do nothing
    def mock_setup_llm_client(self, provider="ollama"):
        self.provider = provider
        self.agent = MagicMock()

    def patched_get_response(self, question: str, context: str = ""):
        nest_asyncio.apply()
        return f"Mocked response to: {question}"

    def patched_structured_get_response(
        self, question: str, context: str = "", response_model=None
    ):
        nest_asyncio.apply()
        if response_model:
            # Return a mock instance of the response model
            return MagicMock(spec=response_model)
        return f"Mocked structured response to: {question}"

    # Patch the methods
    LangModel._setup_llm_client = mock_setup_llm_client
    LangModel._fetch_provider_models = lambda self, provider: ["llama3.2", "qwen3"]
    LangModel.get_response = patched_get_response
    StructuredLangModel.get_response = patched_structured_get_response

    yield
