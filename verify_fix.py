import asyncio
from base_agent.llminterface import LangModel
import nest_asyncio

nest_asyncio.apply()

async def test():
    print("Initializing LangModel...")
    # This might fail if Ollama is not running, but we want to see if it hits TypeError
    lm = LangModel(model='llama3.2', provider='ollama')
    print("Calling get_response...")
    try:
        # Mocking the agent.run to avoid network calls but check arguments
        from unittest.mock import MagicMock
        lm.agent = MagicMock()
        # We need to mock the coroutine
        async def mock_run(*args, **kwargs):
            print(f"Agent.run called with kwargs: {list(kwargs.keys())}")
            class Data: data = "mock response"
            return Data()
        lm.agent.run = mock_run
        
        lm.get_response("hello", context="some context")
        print("Success: get_response called without TypeError")
    except TypeError as e:
        print(f"FAILED with TypeError: {e}")
    except Exception as e:
        print(f"Other exception: {e}")

if __name__ == "__main__":
    asyncio.run(test())
