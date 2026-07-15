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
        self,
        collection_name: str = "",
        new_model: str = "",
        batch_size: int = 100,
        rechunk: bool = True,
        new_chunk_size: int = 1500,
        new_chunk_overlap: int = 200,
    ):
        """Re-embed documents with a new embedding model.

        Use --rechunk to reconstruct source documents from existing chunks,
        re-split with a new chunk size, and write to a shadow collection
        (original collection stays queryable during migration).
        """
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
        print(f"Batch size: {batch_size}")
        if rechunk:
            print(f"Rechunk: enabled")
            print(f"New chunk size: {new_chunk_size}")
            print(f"New chunk overlap: {new_chunk_overlap}")
        print()

        stats = self.DE.reembed(
            collection_name=collection_name,
            new_model=model_to_use,
            batch_size=batch_size,
            rechunk=rechunk,
            new_chunk_size=new_chunk_size,
            new_chunk_overlap=new_chunk_overlap,
        )

        if rechunk:
            print(f"\nRe-chunk + re-embed complete!")
            print(f"  Old chunks: {stats.get('total_old_chunks', 0)}")
            print(f"  New chunks: {stats.get('total_new_chunks', 0)}")
            print(f"  Old model: {stats['old_model']}")
            print(f"  New model: {stats['new_model']}")
            print(f"  Old chunk size: {stats.get('old_chunk_size', 'N/A')}")
            print(f"  New chunk size: {stats.get('new_chunk_size', 'N/A')}")
            if stats.get("shadow_collection"):
                print(f"\n  Shadow collection: '{stats['shadow_collection']}'")
                print("  Original collection is untouched and still queryable.")
                print("  To swap:")
                print(f"    1. Verify quality by querying '{stats['shadow_collection']}'")
                print(f"    2. Rename: use rename_collection to swap names")
                print(f"    3. Drop old collection when satisfied")
        else:
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

    def wiki_consolidate(self, collection_name: str = "main"):
        """Merge per-part source pages into one page per original document."""
        wiki = self._get_wiki(collection_name)
        result = wiki.consolidate_part_pages()

        print(f"Wiki Consolidation for '{collection_name}':")
        print(f"  Document groups merged: {result['groups_merged']}")
        print(f"  Part pages removed:     {result['pages_removed']}")
        print(f"  Links rewritten:        {result['links_rewritten']}")
        if not result["groups_merged"]:
            print("  No per-part pages found to consolidate.")
        return result

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

    def list_backends(self):
        """List available database backends and their configuration status."""
        backends = self.DE.list_backends()
        current = [b for b in backends if b["is_current"]][0]["name"] if any(b["is_current"] for b in backends) else "unknown"

        print("Available Database Backends:")
        print(f"  Current: {current}\n")
        for b in backends:
            marker = " ← current" if b["is_current"] else ""
            configured = "configured" if b["is_configured"] else "not configured"
            print(f"  {b['display_name']:12} ({b['name']})  {configured}{marker}")
            if b["is_configured"] and b["location"]:
                print(f"  {'':14}{b['location']}")
        return backends

    def migrate_backend(
        self,
        target_backend: str = "",
        collection_name: str = "",
        batch_size: int = 1000,
        dry_run: bool = False,
        resume: bool = False,
    ):
        """Migrate embeddings to a different database backend.

        Target backend must be pre-configured in .env (PGURL, TARGET_DUCKDB_PATH, TARGET_SQLITE_PATH).
        Use list_backends to see available targets.
        """
        if not target_backend:
            print("Error: --target_backend is required. Choose from: postgresql, duckdb, sqlite")
            print("Use 'list_backends' to see configured backends.")
            return

        backends = self.DE.list_backends()
        current = next((b for b in backends if b["is_current"]), None)
        target = next((b for b in backends if b["name"] == target_backend), None)

        if not target:
            print(f"Error: Unknown backend '{target_backend}'. Choose from: postgresql, duckdb, sqlite")
            return

        if target["is_current"]:
            print(f"Error: '{target_backend}' is the current backend. Choose a different one.")
            return

        if not target["is_configured"]:
            print(f"Error: '{target_backend}' is not configured. Set the appropriate env variable:")
            if target_backend == "postgresql":
                print("  Set PGURL in .env or environment")
            elif target_backend == "duckdb":
                print("  Set TARGET_DUCKDB_PATH in .env or environment")
            elif target_backend == "sqlite":
                print("  Set TARGET_SQLITE_PATH in .env or environment")
            return

        print(f"Migrating from {current['name']} → {target_backend}")
        if collection_name:
            print(f"  Collection: {collection_name}")
        else:
            print("  All collections")
        print(f"  Batch size: {batch_size}")
        if dry_run:
            print("  DRY RUN — no changes will be made")
        if resume:
            print("  Resume — skipping already-migrated records")
        print()

        stats = self.DE.migrate_backend(
            target_backend=target_backend,
            collection_name=collection_name,
            batch_size=batch_size,
            dry_run=dry_run,
            resume=resume,
        )

        if stats["success"]:
            print(f"\nMigration complete!")
            print(f"  Total: {stats['total']}")
            print(f"  Migrated: {stats['migrated']}")
            print(f"  Skipped: {stats['skipped']}")
            print(f"  Source: {stats['source_backend']} ({stats['source_dimension']}-dim)")
            print(f"  Target: {stats['target_backend']} ({stats['target_dimension']}-dim)")
        else:
            print(f"\nMigration failed:")
            for err in stats["errors"]:
                print(f"  - {err}")

        return stats


def main(corpus_path="."):
    fire.Fire(LibbyInterface)
