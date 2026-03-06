import fire
from libbydbot.brain import embed
import pathlib
import os
import fitz
from fitz import EmptyFileError
from glob import glob
from libbydbot.brain import LibbyDBot
from libbydbot.settings import Settings

try:
    settings = Settings()
except Exception as e:
    print(f"Error loading settings: {e}")
    settings = None


class LibbyInterface(LibbyDBot):
    @staticmethod
    def load_available_models():
        if settings is None:
            return {}
        return {name: details["code"] for name, details in settings.models.items()}

    def __init__(
        self,
        name: str = "Libby D. Bot",
        collection_name: str = "Libby D. Bot",
        languages=["pt_BR", "en"],
        model: str = "llama3.2",
        dburl: str = "postgresql://libby:libby123@localhost:5432/libby",
        embed_db: str = "postgresql://libby:libby123@localhost:5432/libby",
    ):
        super().__init__(
            name=name, languages=languages, model=model, dburl=dburl, embed_db=embed_db
        )
        if collection_name != name:
            self.DE.collection_name = collection_name

    def embed(self, corpus_path: str = "."):
        """
        Embed a corpus of documents
        :param corpus_path: path to a folder containing PDFs
        :return:
        """
        print("Processing your corpus using granular chunking...")
        self.DE.embed_path(corpus_path)
        return self.DE

    def answer(self, question: str, collection_name: str = "main"):
        """
        Answer a question based on a collection of documents.
        Prioritizes agentic tool-calling if supported by the base agent.
        :param question: Users question
        :param collection_name: collection of documents on which to base the answer
        :return: Answer to the question
        """
        # If the agent has tools registered, we can try to let it handle retrieval
        # In Pydantic AI, tools are often in _functions
        has_tools = hasattr(self.llm, "agent") and (
            hasattr(self.llm.agent, "tools")
            or hasattr(self.llm.agent, "_functions")
            or hasattr(self.llm.agent, "tool")
        )

        if has_tools:
            self.set_prompt(
                f"You are Libby D. Bot, a research Assistant. Use the search_library tool to find information if needed."
            )
            # Clear manual context to encourage tool use
            self.set_context("")
            return self.ask(question)

        # Fallback to manual RAG
        context = self.DE.retrieve_docs(
            question, collection=collection_name, num_docs=5
        )
        self.set_prompt(f"You are Libby D. Bot, a research Assistant")
        self.set_context(context)

        response = self.ask(question)
        return response

    def generate(self, prompt: str = "", output_file: str = "", prompt_file: str = ""):
        """
        Generate text based on a prompt
        :param prompt: The prompt to generate text from
        :param output_file: Optional file path to save the generated text
        :param prompt_file: Optional file path to read the prompt from
        :return: Generated text
        """
        if not prompt:
            if prompt_file:
                try:
                    with open(prompt_file, "r") as f:
                        prompt = f.read().strip()
                except FileNotFoundError:
                    print(f"Error: Prompt file '{prompt_file}' not found.")
                    return
                except Exception as e:
                    print(f"Error reading prompt file: {e}")
                    return
            else:
                prompt = input("Enter a prompt: ")
        # DE = embed.DocEmbedder("embedding")
        context = self.DE.retrieve_docs(prompt, num_docs=100)
        self.set_prompt("You are Libby D. Bot, a creative and competent writer.")
        self.set_context(context)
        response = self.ask(prompt)

        if output_file:
            with open(output_file, "w") as f:
                f.write(response)
            print(f"Generated text saved to: {output_file}")

        return response

    def reembed(
        self, collection_name: str = "", new_model: str = "", batch_size: int = 100
    ):
        """
        Re-embed documents with a new embedding model.

        :param collection_name: Collection to re-embed (empty = all collections)
        :param new_model: New embedding model (empty = use settings default)
        :param batch_size: Batch size for processing (default: 100)
        :return: Statistics about the re-embedding process
        """
        # First, run migration to add embedding_model column if needed
        print("Checking for database schema updates...")
        self.DE._migrate_add_embedding_model()

        # Show current model info
        print("\nCurrent embedding models in database:")
        info = self.DE.get_embedding_model_info()
        for model, collections in info.get("models", {}).items():
            for coll, count in collections.items():
                print(f"  - {model}: {count} documents in '{coll}'")
        print(f"Total documents: {info.get('total_documents', 0)}")

        # Determine new model
        model_to_use = new_model if new_model else None

        print(f"\nRe-embedding with model: {model_to_use or 'default from settings'}")
        if collection_name:
            print(f"Collection: {collection_name}")
        else:
            print("Collection: all")
        print(f"Batch size: {batch_size}")
        print()

        stats = self.DE.reembed(
            collection_name=collection_name,
            new_model=model_to_use,
            batch_size=batch_size,
        )

        print(f"\nRe-embedding complete!")
        print(f"  Total documents: {stats['total']}")
        print(f"  Updated: {stats['updated']}")
        print(f"  Old model: {stats['old_model']}")
        print(f"  New model: {stats['new_model']}")
        if stats["errors"]:
            print(f"  Errors: {len(stats['errors'])}")
            for err in stats["errors"][:5]:
                print(f"    - {err}")
            if len(stats["errors"]) > 5:
                print(f"    ... and {len(stats['errors']) - 5} more errors")

        if "backup_table" in stats and stats["backup_table"]:
            print(f"\n  Backup table: {stats['backup_table']}")
            print(f"  The backup table has been preserved for safety.")
            print(
                f"  You can drop it manually after verifying the re-embedding was successful."
            )

        return stats

    def model_info(self):
        """
        Show information about embedding models used in the database.
        """
        # First, run migration to add embedding_model column if needed
        self.DE._migrate_add_embedding_model()

        info = self.DE.get_embedding_model_info()

        print("Embedding Model Information:")
        print(f"  Total documents: {info.get('total_documents', 0)}")
        print()
        for model, collections in info.get("models", {}).items():
            print(f"  Model: {model}")
            for coll, count in collections.items():
                print(f"    - Collection '{coll}': {count} documents")

        return info


def main(corpus_path="."):
    fire.Fire(LibbyInterface)
