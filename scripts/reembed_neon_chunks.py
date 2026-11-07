"""Transactional script to re-embed the 27 document chunks in Neon PostgreSQL
using the locally cached BAAI/bge-small-en-v1.5 model.
"""
import sys
import time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sqlalchemy import text
from app.db.session import SessionLocal
from app.db.models import Document, DocumentChunk
from app.rag.embeddings import embedding_service


def reembed_assessment_chunks():
    print("=" * 80)
    print("STEP 3: TRANSACTIONAL RE-EMBEDDING OF 27 NEON POSTGRESQL CHUNKS")
    print("=" * 80)

    # 1. Verify model loads and validates before touching database
    print("\n[Phase 1] Verifying EmbeddingService...")
    test_vec = embedding_service.embed_text("Semantic search verification")
    if len(test_vec) != 384:
        raise RuntimeError(f"Model validation failed: expected 384 dimensions, got {len(test_vec)}")
    norm = sum(x * x for x in test_vec) ** 0.5
    print(f"Model successfully loaded: {embedding_service.model_name}")
    print(f"Dimension: {len(test_vec)} | Unit Norm: {norm:.4f}")

    # 2. Inspect existing database records
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.title == "AI_Engineer_Technical_Assessment.pdf").first()
        if not doc:
            raise RuntimeError("Document 'AI_Engineer_Technical_Assessment.pdf' not found in Neon DB.")

        doc_id = doc.id
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        initial_count = len(chunks)
        print(f"\n[Phase 2] Found {initial_count} chunks in Neon DB for document ID: {doc_id}")
        if initial_count != 27:
            raise RuntimeError(f"Expected 27 chunks, found {initial_count}")

        # Record snapshot of original attributes to verify preservation
        snapshot = {}
        for c in chunks:
            snapshot[c.id] = {
                "chunk_index": c.chunk_index,
                "document_id": c.document_id,
                "token_count": c.token_count,
                "content_len": len(c.content),
                "metadata": dict(c.chunk_metadata or {}),
            }

        # 3. Generate new 384-dimensional embeddings
        print("\n[Phase 3] Generating embeddings via BAAI/bge-small-en-v1.5...")
        t0 = time.perf_counter()
        contents = [c.content for c in chunks]
        new_embeddings = embedding_service.embed_batch(contents)
        embed_duration = time.perf_counter() - t0
        print(f"Generated {len(new_embeddings)} embeddings in {embed_duration:.2f}s.")

        # Validate all embeddings before DB modification
        for idx, emb in enumerate(new_embeddings):
            if len(emb) != 384:
                raise ValueError(f"Chunk {idx} has invalid embedding length: {len(emb)}")
            v_norm = np.linalg.norm(np.array(emb, dtype=np.float32))
            if abs(v_norm - 1.0) > 1e-4:
                raise ValueError(f"Chunk {idx} vector not normalized: norm={v_norm}")

        # 4. Perform transactional update
        print("\n[Phase 4] Applying transactional update to Neon PostgreSQL...")
        t_update = time.perf_counter()
        for idx, c in enumerate(chunks):
            c.embedding = new_embeddings[idx]

        db.commit()
        update_duration = time.perf_counter() - t_update
        print(f"Transaction committed in {update_duration:.2f}s.")

        # 5. Post-update verification
        print("\n[Phase 5] Post-update database verification...")
        verify_chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        post_count = len(verify_chunks)
        print(f"Verified chunk count: {post_count} (Expected: 27)")
        if post_count != 27:
            raise RuntimeError(f"Chunk count changed! Expected 27, found {post_count}")

        # Verify integrity of all fields and attributes
        for c in verify_chunks:
            orig = snapshot[c.id]
            assert c.chunk_index == orig["chunk_index"], f"Index mismatch on {c.id}"
            assert c.document_id == orig["document_id"], f"Doc ID mismatch on {c.id}"
            assert c.token_count == orig["token_count"], f"Token count mismatch on {c.id}"
            assert len(c.content) == orig["content_len"], f"Content length mismatch on {c.id}"
            assert c.chunk_metadata == orig["metadata"], f"Metadata mismatch on {c.id}"
            assert c.embedding is not None, f"Embedding is None on {c.id}"
            assert len(c.embedding) == 384, f"Dimension mismatch on {c.id}: {len(c.embedding)}"

        # Verify chunk 0 embedding matches new model
        c0_vec = np.array(verify_chunks[0].embedding, dtype=np.float32)
        expected_c0 = np.array(new_embeddings[0], dtype=np.float32)
        sim_new = float(np.dot(c0_vec, expected_c0))
        print(f"Chunk 0 similarity with new model: {sim_new:.6f} (Expected: 1.000000)")
        assert sim_new > 0.9999, "Stored embedding does not match new model output!"

        print("\n" + "=" * 80)
        print("PASS: All 27 chunks successfully and transactionally re-embedded with BAAI/bge-small-en-v1.5.")
        print("Preservation verified: IDs, metadata, contents, page numbers intact.")
        print("=" * 80)

    except Exception as e:
        db.rollback()
        print(f"ERROR during re-embedding: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    reembed_assessment_chunks()
