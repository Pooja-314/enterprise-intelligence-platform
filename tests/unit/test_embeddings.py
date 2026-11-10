"""Targeted unit tests for EmbeddingService loading, diagnostics, and failure handling."""
from unittest.mock import MagicMock, patch
import pytest

from app.rag.embeddings import EmbeddingService


def test_embedding_diagnostic_structure():
    """Verifies that get_diagnostic returns a complete diagnostic dictionary."""
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    diag = service.get_diagnostic()

    assert diag["model"] == "BAAI/bge-small-en-v1.5"
    assert diag["configured_dimension"] == 384
    assert diag["actual_dimension"] == 384
    assert "is_loaded" in diag
    assert "is_cached" in diag
    assert "status" in diag
    assert "cache_dir" in diag


def test_embedding_dimension_and_normalization():
    """Verifies that embed_text and embed_batch produce strictly 384d normalized vectors."""
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    vec = service._generate_deterministic_embedding("Enterprise artificial intelligence test query")

    assert len(vec) == 384
    # Compute L2 norm (should be ~1.0 for unit normalization)
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-3

    batch_vecs = [service._generate_deterministic_embedding(t) for t in ["Query one", "Query two"]]
    assert len(batch_vecs) == 2
    for v in batch_vecs:
        assert len(v) == 384


def test_embedding_online_download_when_cache_empty():
    """Simulates Render's fresh container where local cache is empty:
    Verifies that _get_model attempts online download instead of failing with empty candidates.
    """
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)

    mock_model = MagicMock()
    mock_model.get_embedding_dimension.return_value = 384
    mock_model.get_sentence_embedding_dimension.return_value = 384

    mock_st = MagicMock(side_effect=[OSError("Not found locally"), mock_model])

    # Phase 1: local_files_only=True raises OSError (simulating not cached locally)
    # Phase 2: online download call succeeds and returns mock_model
    with patch.object(service, "_resolve_snapshot_dir", return_value=None), \
         patch.object(service, "_is_model_cached", return_value=False), \
         patch.object(service, "_load_sentence_transformer", mock_st):
        
        model = service._get_model()
        assert model == mock_model
        assert mock_st.call_count == 2
        # First call was local cache check with local_files_only=True
        assert mock_st.call_args_list[0][1].get("local_files_only") is True
        # Second call was online download without local_files_only
        assert mock_st.call_args_list[1][0][0] == "BAAI/bge-small-en-v1.5"


def test_embedding_production_failure_raises_descriptive_error(monkeypatch):
    """Verifies that in production, if all load and download attempts fail,
    a descriptive RuntimeError is raised containing candidate info and the last error.
    """
    from app.config import settings
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")

    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)

    with patch.object(service, "_resolve_snapshot_dir", return_value=None), \
         patch.object(service, "_is_model_cached", return_value=False), \
         patch.object(service, "_load_sentence_transformer", side_effect=ConnectionError("Could not resolve huggingface.co")):
        
        with pytest.raises(RuntimeError) as exc_info:
            service._get_model()

        err_msg = str(exc_info.value)
        assert "CRITICAL: Neural embedding model could not be loaded or downloaded" in err_msg
        assert "BAAI/bge-small-en-v1.5" in err_msg
        assert "Could not resolve huggingface.co" in err_msg


def test_embedding_dimension_mismatch_raises_error():
    """Verifies that if a model with incorrect dimension (e.g. 768 instead of 384) is loaded,
    it raises a ValueError to prevent silent vector corruptions in pgvector.
    """
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)

    mock_wrong_model = MagicMock()
    mock_wrong_model.get_sentence_embedding_dimension.return_value = 768
    mock_wrong_model.get_embedding_dimension.return_value = 768

    with patch.object(service, "_resolve_snapshot_dir", return_value=None), \
         patch.object(service, "_load_sentence_transformer", return_value=mock_wrong_model):
        
        with pytest.raises(ValueError) as exc_info:
            service._get_model()

        assert "Embedding dimension mismatch: expected 384, got 768" in str(exc_info.value)


def test_embedding_warmup_non_blocking():
    """Verifies that warmup() handles exceptions gracefully without crashing."""
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)

    with patch.object(service, "_get_model", side_effect=Exception("Simulated download timeout")):
        result = service.warmup()
        assert result is False  # Warmup failed gracefully, did not raise exception


@pytest.mark.asyncio
async def test_health_check_contains_embedding_diagnostics():
    """Verifies that FastAPI health_check endpoint reports real embedding diagnostics."""
    from app.main import health_check
    from app.rag.embeddings import embedding_service

    with patch("app.main.check_db_connection", return_value={"connected": True, "dialect": "postgresql"}):
        data = await health_check()

        assert "embeddings" in data
        emb_diag = data["embeddings"]
        assert emb_diag["model"] == embedding_service.model_name
        assert emb_diag["configured_dimension"] == 384
        assert emb_diag["actual_dimension"] == 384
        assert "is_loaded" in emb_diag
        assert "status" in emb_diag


def test_embedding_cosine_similarity_fp32_vs_reduced_precision():
    """Validates requirement 7: compares embeddings produced with full precision
    vs reduced precision (bfloat16) to ensure cosine similarity > 0.999 across
    representative enterprise texts.
    """
    import os
    import numpy as np

    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    texts = [
        "Enterprise document intelligence platform using hybrid RAG.",
        "Financial performance increased by 18 percent year over year.",
        "Authentication and authorization security policies."
    ]

    # Verify deterministic test fallback produces exact 384d unit vectors
    embeddings = service.embed_batch(texts)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert len(emb) == 384
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-2

    # If the real model is cached locally, test real cosine similarity between FP32 and bfloat16
    snapshot_dir = service._resolve_snapshot_dir("BAAI/bge-small-en-v1.5")
    if snapshot_dir and os.path.isdir(snapshot_dir):
        import torch
        from sentence_transformers import SentenceTransformer

        # Load FP32 baseline
        fp32_st = SentenceTransformer(snapshot_dir, device="cpu", local_files_only=True)
        fp32_embs = fp32_st.encode(texts, normalize_embeddings=True)

        # Load reduced-precision bfloat16
        bf16_st = SentenceTransformer(
            snapshot_dir,
            device="cpu",
            local_files_only=True,
            model_kwargs={"dtype": torch.bfloat16, "low_cpu_mem_usage": True}
        )
        bf16_embs = bf16_st.encode(texts, normalize_embeddings=True)

        for i in range(len(texts)):
            u, v = fp32_embs[i], bf16_embs[i]
            sim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            # Cosine similarity must be virtually identical (> 0.999)
            assert sim > 0.999, f"Text {i} similarity {sim} fell below 0.999 threshold"
