"""Focused End-to-End Diagnostic of the exact request path used by the browser.
Sanitized output: Never prints database URLs, passwords, API keys, or secrets.
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import text
from app.config import settings
from app.db.session import SessionLocal
from app.db.models import Document, DocumentChunk, User, Conversation, Message
from app.agent.query_router import conversation_router, QueryIntent
from app.rag.generator import rag_generator
from app.rag.retrieval import HybridRetriever
from app.rag.embeddings import embedding_service


def run_diagnostic():
    print("=" * 80)
    print("STAGE 6 AI CHAT - LIVE END-TO-END BROWSER REQUEST DIAGNOSTIC")
    print("=" * 80)

    query = (
        "Explain the five core modules required by the technical assessment. "
        "For each module, describe its key functional requirements and cite the relevant source document sections or pages."
    )
    print(f"\n[1] Target User Query:\n\"{query}\"")

    # 1. Intent Classification
    intent = conversation_router.classify_intent(query)
    print(f"\n[2] Query Classification & Route:")
    print(f" - Classified Intent: {intent.value}")
    print(f" - Target Route: {'_handle_knowledge_base_rag (Direct RAG)' if intent == QueryIntent.KNOWLEDGE_BASE_RAG else intent.value}")

    # 2. Database & Chunks Inspection
    db = SessionLocal()
    try:
        dialect = db.bind.dialect.name if db.bind else "unknown"
        print(f"\n[3] Database Connection:")
        print(f" - Dialect: {dialect} (is_postgres: {dialect == 'postgresql'})")

        docs = db.query(Document).filter(Document.title.ilike("%assessment%")).all()
        print(f" - Assessment Documents in DB: {len(docs)}")
        target_doc = None
        for d in docs:
            chunk_count = db.query(DocumentChunk).filter(DocumentChunk.document_id == d.id).count()
            print(f"   • Doc ID: {d.id} | Title: {d.title} | Status: {d.processing_status} | Chunks: {chunk_count}")
            if "technical_assessment" in d.title.lower():
                target_doc = d

        if not target_doc and docs:
            target_doc = docs[0]

        total_chunks_in_db = db.query(DocumentChunk).count()
        print(f" - Total Document Chunks in entire DB: {total_chunks_in_db}")

        # 3. Embedding Model Verification
        print(f"\n[4] Embedding Model Verification:")
        model_name = embedding_service.model_name
        is_neural = embedding_service._model is not None
        print(f" - Configured Model Name: {model_name}")
        t0 = time.perf_counter()
        query_vec = embedding_service.embed_text(query)
        emb_latency = time.perf_counter() - t0
        vec_dim = len(query_vec)
        vec_norm = sum(x * x for x in query_vec) ** 0.5
        print(f" - Neural Model Active: {embedding_service._model is not None}")
        print(f" - Vector Dimension: {vec_dim} (Expected: 384)")
        print(f" - Vector L2 Norm: {vec_norm:.4f} (Expected: 1.0000)")
        print(f" - Query Embedding Latency: {emb_latency:.3f}s")

        # 4. Dense Vector Retrieval
        retriever = HybridRetriever()
        print(f"\n[5] Dense Vector Retrieval (top_k={settings.TOP_K_VECTOR}):")
        dense_results = retriever.vector_search(db, query_vec, top_k=settings.TOP_K_VECTOR)
        print(f" - Retrieved {len(dense_results)} dense chunks:")
        for idx, r in enumerate(dense_results[:8], 1):
            page = r.metadata.get("page_number", "?")
            print(f"   [{idx}] ID: {r.chunk_id[:8]} | Page: {page} | CosSim: {r.dense_score:.4f} | Snippet: {r.content[:75].replace(chr(10), ' ')}")

        # 5. Sparse Keyword Retrieval
        print(f"\n[6] Sparse Keyword Retrieval (top_k={settings.TOP_K_KEYWORD}):")
        sparse_results = retriever.keyword_search(db, query, top_k=settings.TOP_K_KEYWORD)
        print(f" - Retrieved {len(sparse_results)} sparse chunks:")
        for idx, r in enumerate(sparse_results[:8], 1):
            page = r.metadata.get("page_number", "?")
            print(f"   [{idx}] ID: {r.chunk_id[:8]} | Page: {page} | RankScore: {r.sparse_score:.4f} | Snippet: {r.content[:75].replace(chr(10), ' ')}")

        # 6. RRF Fusion
        target_top_k = max(8, getattr(settings, "TOP_K_FINAL", 8))
        fused = retriever.reciprocal_rank_fusion(dense_results, sparse_results, k=settings.RRF_K)
        top_fused = fused[:target_top_k]
        print(f"\n[7] RRF Merged Results (top_k={len(top_fused)}):")
        pages_present = set()
        for idx, r in enumerate(top_fused, 1):
            page = r.metadata.get("page_number", "?")
            pages_present.add(str(page))
            print(f"   [{idx}] ID: {r.chunk_id[:8]} | DenseRank: {r.dense_rank} | SparseRank: {r.sparse_rank} | RRF: {r.rrf_score:.6f} | Page: {page}")
            print(f"       Snippet: {r.content[:95].replace(chr(10), ' ')}")

        print(f"\n[8] Assessment Pages Analysis in Top Chunks:")
        print(f" - Pages in Top Context: {sorted(list(pages_present))}")
        print(f" - Page 2 Present: {'2' in pages_present}")
        print(f" - Page 4 Present: {'4' in pages_present}")

        # 7. Context Construction & LLM Generation
        print(f"\n[9] Running RAG Generator on Live Database:")
        rag_res = rag_generator.query(db, query, top_k=target_top_k)
        print(f" - Provider: {rag_res.provider} | Model: {rag_res.model}")
        print(f" - Insufficient Evidence Flag: {rag_res.is_insufficient_evidence}")
        print(f" - Citations Count: {len(rag_res.citations)}")
        for c in rag_res.citations:
            print(f"   • {c.document_title} | Page: {c.page_number} | Chunk: {c.chunk_id[:8]}")
        print(f" - Verification Confidence: {rag_res.verification.confidence_score * 100:.1f}%")
        print(f" - Verification Claims Grounded: {rag_res.verification.supported_claims}/{rag_res.verification.total_claims}")
        print(f" - Hallucination Detected: {rag_res.verification.hallucination_detected}")
        print(f"\n--- GENERATED ANSWER SNIPPET ---")
        answer_lines = rag_res.answer.strip().split("\n")
        for line in answer_lines[:15]:
            print(f" | {line}")
        if len(answer_lines) > 15:
            print(f" | ... [{len(answer_lines) - 15} more lines]")

        # 8. Full route_and_execute Simulation
        print(f"\n[10] Running conversation_router.route_and_execute():")
        routed_res = conversation_router.route_and_execute(
            db=db,
            query=query,
            user_id=1,
            conversation_id="test_diag_conv",
            conversation_context="",
            max_steps=5,
        )
        print(f" - Status: {routed_res.status}")
        print(f" - Intent: {routed_res.intent.value}")
        print(f" - Tools Used: {routed_res.tools_used}")
        print(f" - Citations Count: {len(routed_res.citations)}")
        print(f" - Answer Starts With: {routed_res.answer[:80]}...")
        is_insufficient = "INSUFFICIENT_EVIDENCE" in routed_res.answer.upper()
        print(f" - Contains 'INSUFFICIENT_EVIDENCE': {is_insufficient}")

    finally:
        db.close()


if __name__ == "__main__":
    run_diagnostic()
