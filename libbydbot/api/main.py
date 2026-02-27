import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from libbydbot.api.schemas import HealthResponse
from libbydbot.api.routes import embed, retrieve
from libbydbot.brain.embed import DocEmbedder


@dataclass
class AppState:
    embedder: DocEmbedder | None = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_state

    dburl = os.getenv("EMBED_DB", "duckdb:///data/embeddings.duckdb")
    embedding_model = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

    logger.info(f"Initializing DocEmbedder with dburl={dburl}, model={embedding_model}")

    col_name = os.getenv("COLLECTION_NAME", "main")
    embedder = DocEmbedder(
        col_name=col_name, dburl=dburl, embedding_model=embedding_model
    )

    app_state.embedder = embedder

    logger.info("DocEmbedder initialized successfully")

    yield

    if embedder._connection:
        try:
            embedder._connection.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Libby D. Bot API",
        description="REST API for Libby D. Bot - AI-powered librarian for RAG document embedding and retrieval",
        version="0.6.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(embed.router, prefix="/api")
    app.include_router(retrieve.router)

    @app.get("/api/health", response_model=HealthResponse, tags=["health"])
    def health_check():
        """Health check endpoint."""
        if app_state.embedder is None:
            return HealthResponse(status="unhealthy", database="none", version="0.6.0")
        db_type = (
            "duckdb"
            if "duckdb" in app_state.embedder.dburl
            else "sqlite"
            if "sqlite" in app_state.embedder.dburl
            else "postgresql"
        )
        return HealthResponse(status="healthy", database=db_type, version="0.6.0")

    @app.get("/", tags=["root"])
    def root():
        """Root endpoint with API information."""
        return {
            "name": "Libby D. Bot API",
            "version": "0.6.0",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()


def run():
    """Entry point for the libby-server CLI command."""
    import argparse

    parser = argparse.ArgumentParser(description="Libby D. Bot API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    args = parser.parse_args()

    uvicorn.run(
        "libbydbot.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    run()
