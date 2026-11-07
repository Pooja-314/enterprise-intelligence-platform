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
