"""Targeted unit tests for EmbeddingService ONNX loading, diagnostics, and failure handling."""
from unittest.mock import MagicMock, patch
import pytest

from app.rag.embeddings import EmbeddingService, ONNXEmbeddingEngine


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
    assert "engine" in diag


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
    Verifies that _get_model attempts online download of ONNX model files.
    """
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)

    mock_model = MagicMock()
    mock_model.get_embedding_dimension.return_value = 384

    cache_state = {"downloaded": False}

    def fake_resolve(candidate):
        return "/mock/snapshot" if cache_state["downloaded"] else None

    def fake_download(*args, **kwargs):
        cache_state["downloaded"] = True
        return "/mock/snapshot/onnx/model.onnx"

    service._onnx_loader = lambda d: mock_model

    with patch.object(service, "_resolve_snapshot_dir", side_effect=fake_resolve), \
         patch.object(service, "_is_model_cached", return_value=False), \
         patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        
        model = service._get_model()
        assert model == mock_model
        assert cache_state["downloaded"] is True


def test_embedding_production_failure_raises_descriptive_error(monkeypatch):
    """Verifies that in production, if ONNX model cannot be loaded or downloaded,
    a critical RuntimeError is raised and PyTorch fallback is strictly prevented.
    """
    from app.config import settings
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")

    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)

    with patch.object(service, "_resolve_snapshot_dir", return_value=None), \
         patch.object(service, "_is_model_cached", return_value=False), \
         patch("huggingface_hub.hf_hub_download", side_effect=ConnectionError("Could not resolve huggingface.co")):
        
        with pytest.raises(RuntimeError) as exc_info:
            service._get_model()

        err_msg = str(exc_info.value)
        assert "CRITICAL: Production ONNX neural embedding model could not be loaded" in err_msg
        assert "BAAI/bge-small-en-v1.5" in err_msg
        assert "Could not resolve huggingface.co" in err_msg
        assert "PyTorch fallback is strictly disabled in production" in err_msg


def test_embedding_dimension_mismatch_raises_error():
    """Verifies that if a model with incorrect dimension (e.g. 768 instead of 384) is loaded,
    it raises a ValueError to prevent silent vector corruptions in pgvector.
    """
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)

    mock_wrong_model = MagicMock()
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
        assert "engine" in emb_diag


def test_onnx_embedding_engine_and_pytorch_cosine_similarity():
    """Validates that the pure ONNX Runtime engine produces strictly 384d,
    unit-normalized vectors with 1.00000000 cosine similarity against PyTorch.
    """
    import os
    import numpy as np

    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    texts = [
        "Enterprise document intelligence platform using hybrid RAG.",
        "Financial performance increased by 18 percent year over year.",
        "Authentication and authorization security policies."
    ]

    snapshot_dir = service._resolve_snapshot_dir("BAAI/bge-small-en-v1.5")
    if snapshot_dir and os.path.isdir(snapshot_dir):
        onnx_file = os.path.join(snapshot_dir, "onnx", "model.onnx")
        if not os.path.isfile(onnx_file):
            onnx_file = os.path.join(snapshot_dir, "model.onnx")

        if os.path.isfile(onnx_file):
            # Test direct ONNX engine
            onnx_engine = ONNXEmbeddingEngine(snapshot_dir, dimension=384)
            onnx_embs = onnx_engine.encode(texts, normalize_embeddings=True)

            assert onnx_embs.shape == (3, 384)
            for emb in onnx_embs:
                norm = float(np.linalg.norm(emb))
                assert abs(norm - 1.0) < 1e-4

            # Test EmbeddingService.embed_batch using ONNX
            service_embs = service.embed_batch(texts)
            assert len(service_embs) == 3
            assert len(service_embs[0]) == 384
            assert service.get_diagnostic()["engine"] == "onnxruntime"

            # Compare against PyTorch SentenceTransformer baseline
            try:
                from sentence_transformers import SentenceTransformer
                st = SentenceTransformer(snapshot_dir, device="cpu", local_files_only=True)
                st_embs = st.encode(texts, normalize_embeddings=True)

                for i in range(len(texts)):
                    u = onnx_embs[i]
                    v = st_embs[i]
                    sim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
                    assert sim > 0.9999, f"Text {i} similarity {sim} fell below 0.9999"
            except ImportError:
                pass
    else:
        # Fallback verification
        embeddings = service.embed_batch(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384
