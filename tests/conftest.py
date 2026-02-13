import pytest
from unittest.mock import patch
import numpy as np

@pytest.fixture(autouse=True)
def mock_embeddings():
    # Mocking Ollama/Gemini embeddings globally
    with patch('libbydbot.brain.embed.DocEmbedder._generate_embedding') as mocked:
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
def apply_monkeypatch():
    from base_agent.llminterface import LangModel, StructuredLangModel
    import asyncio
    import nest_asyncio
    
    def patched_get_response(self, question: str, context: str = ""):
        if not self.agent:
            self._setup_llm_client(self.provider or 'ollama')
        nest_asyncio.apply()
        async def _run():
            history = self.chat_history.get_all()
            full_prompt = f"Context: {context}\n\nQuestion: {question}" if context else question
            result = await self.agent.run(full_prompt, message_history=history)
            for msg in result.new_messages():
                self.chat_history.enqueue(msg)
            return result.data
        return asyncio.run(_run())

    def patched_structured_get_response(self, question: str, context: str = "", response_model = None):
        if not self.agent:
            self._setup_llm_client(self.provider or 'ollama')
        nest_asyncio.apply()
        async def _run():
            history = self.chat_history.get_all()
            full_prompt = f"Context: {context}\n\nQuestion: {question}" if context else question
            result = await self.agent.run(full_prompt, message_history=history, result_type=response_model)
            for msg in result.new_messages():
                self.chat_history.enqueue(msg)
            return result.data
        return asyncio.run(_run())

    LangModel.get_response = patched_get_response
    StructuredLangModel.get_response = patched_structured_get_response
    yield
