"""Production wiring: pgvector store + Anthropic client (reads ANTHROPIC_API_KEY / ant auth profile)."""
import os

import anthropic
import psycopg

from .api import create_app
from .retrieval import HashingEmbedder, PgVectorStore

# Swap HashingEmbedder for a hosted embedding model (same Embedder interface) in production.
app = create_app(client=anthropic.Anthropic(), store=PgVectorStore(psycopg.connect(os.environ["DATABASE_URL"]), HashingEmbedder()))
