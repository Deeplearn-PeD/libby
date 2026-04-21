"""
Legacy Fire-based CLI for Libby D. Bot.
Kept for scripting use. The default `libby` command now launches the TUI.
Use `libby-cli` for command-line scripting.
"""

import fire
from libbydbot.brain import embed
import pathlib
import os
import fitz
from fitz import EmptyFileError
from glob import glob
from libbydbot.brain import LibbyDBot
from libbydbot.brain.wiki import WikiManager
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
        """Embed a corpus of documents."""
        print("Processing your corpus using granular chunking...")
        self.DE.embed_path(corpus_path)
        return self.DE

    def answer(self, question: str, collection_name: str = "main"):
        """Answer a question based on a collection of documents."""
        has_tools = hasattr(self.llm, "agent") and (
            hasattr(self.llm.agent, "tools")
            or hasattr(self.llm.agent, "_functions")
            or hasattr(self.llm.agent, "tool")
        )

        if has_tools:
            self.set_prompt(
                "You are Libby D. Bot, a research Assistant. Use the search_library tool to find information if needed."
            )
            self.set_context("")
            return self.ask(question)

        context = self.DE.retrieve_docs(question, collection=collection_name, num_docs=5)
        self.set_prompt("You are Libby D. Bot, a research Assistant")
        self.set_context(context)
        return self.ask(question)

    def generate(self, prompt: str = "", output_file: str = "", prompt_file: str = ""):
        """Generate text based on a prompt."""
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
        """Re-embed documents with a new embedding model."""
        print("Checking for database schema updates...")
        self.DE._migrate_add_embedding_model()

        print("\nCurrent embedding models in database:")
        info = self.DE.get_embedding_model_info()
        for model, collections in info.get("models", {}).items():
            for coll, count in collections.items():
                print(f"  - {model}: {count} documents in '{coll}'")
        print(f"Total documents: {info.get('total_documents', 0)}")

        model_to_use = new_model if new_model else None
        print(f"\nRe-embedding with model: {model_to_use or 'default from settings'}")
        print(f"Collection: {collection_name or 'all'}")
        print(f"Batch size: {batch_size}\n")

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

        if stats.get("backup_table"):
            print(f"\n  Backup table: {stats['backup_table']}")
            print("  The backup table has been preserved for safety.")
            print("  You can drop it manually after verifying the re-embedding was successful.")

        return stats

    def model_info(self):
        """Show information about embedding models used in the database."""
        self.DE._migrate_add_embedding_model()
        info = self.DE.get_embedding_model_info()

        print("Embedding Model Information:")
        print(f"  Total documents: {info.get('total_documents', 0)}")
        for model, collections in info.get("models", {}).items():
            print(f"  Model: {model}")
            for coll, count in collections.items():
                print(f"    - Collection '{coll}': {count} documents")
        return info

    def _get_wiki(self, collection_name: str) -> WikiManager:
        """Return a WikiManager for the given collection."""
        wiki_base = settings.wiki_base_path if settings else ""
        return WikiManager(
            collection_name=collection_name,
            wiki_base=wiki_base,
            model=self.model,
        )

    def wiki_ingest(self, corpus_path: str = ".", collection_name: str = "main"):
        """Ingest documents from a path into the wiki."""
        wiki = self._get_wiki(collection_name)
        results = []

        for pdf_path in glob(os.path.join(corpus_path, "*.pdf")):
            try:
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc_name = doc.metadata.get("title", os.path.basename(pdf_path))
            except Exception as e:
                print(f"Error reading {pdf_path}: {e}")
                continue

            result = wiki.ingest_source(doc_name, text)
            results.append(result)
            print(f"  Ingested: {doc_name} ({result['pages_touched']} pages touched)")

        if not results:
            print("No PDF documents found to ingest into wiki.")
        else:
            total_pages = sum(r["pages_touched"] for r in results)
            print(f"\nWiki ingest complete: {len(results)} sources, {total_pages} pages touched.")
        return results

    def wiki_query(self, question: str, collection_name: str = "main", file_answer: bool = False):
        """Query the wiki and synthesize an answer."""
        wiki = self._get_wiki(collection_name)
        result = wiki.query(question, file_answer=file_answer)

        print(f"\nConfidence: {result['confidence']}")
        print(f"Sources used: {', '.join(result['sources_used']) or 'None'}")
        if result.get("gaps"):
            print(f"Gaps: {', '.join(result['gaps'])}")
        print(f"\n{result['answer']}")
        if result.get("filed_page"):
            print(f"\nAnswer filed to wiki page: {result['filed_page']}")
        return result

    def wiki_lint(self, collection_name: str = "main", auto_fix: bool = False):
        """Health-check the wiki."""
        wiki = self._get_wiki(collection_name)
        report = wiki.lint(auto_fix=auto_fix)

        print(f"Wiki Lint Report for '{collection_name}':")
        print(f"  Orphan pages: {len(report['orphan_pages'])}")
        print(f"  Broken links: {len(report['broken_links'])}")
        print(f"  Contradictions: {len(report['contradictions'])}")
        print(f"  Stale claims: {len(report['stale_claims'])}")
        print(f"  Missing pages: {len(report['missing_pages'])}")
        if report.get("fixes_applied"):
            print(f"  Fixes applied: {report['fixes_applied']}")
        if report.get("suggestions"):
            print("\nSuggestions:")
            for s in report["suggestions"]:
                print(f"  - {s}")
        return report

    def wiki_status(self, collection_name: str = "main"):
        """Show wiki statistics."""
        wiki = self._get_wiki(collection_name)
        status = wiki.status()

        print(f"Wiki Status for '{collection_name}':")
        print(f"  Path: {status['wiki_path']}")
        print(f"  Total pages: {status['total_pages']}")
        for category, count in status["page_counts"].items():
            print(f"    {category}: {count}")
        print(f"  Orphan pages: {status['orphan_pages']}")
        print(f"  Broken links: {status['broken_links']}")
        if status["last_operation"]:
            print(f"  Last operation: {status['last_operation']}")
        return status


def main(corpus_path="."):
    fire.Fire(LibbyInterface)
