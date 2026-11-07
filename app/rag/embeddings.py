import hashlib
import logging
import os
from typing import List, Optional
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating dense vector embeddings with SentenceTransformers.
    
    Loads locally cached 384-dimensional neural models (specifically BAAI/bge-small-en-v1.5)
    with unit normalization and dimension validation.
    """

    def __init__(self, model_name: Optional[str] = None, dimension: Optional[int] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self._model = None
        self._load_attempted = False

    def _is_model_cached(self, model_name: str) -> bool:
        """Checks if model files are cached locally to avoid slow network timeouts."""
        hub_name = "models--" + model_name.replace("/", "--")
        cache_path = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", hub_name))
        return os.path.isdir(cache_path)

    def _resolve_snapshot_dir(self, model_name: str) -> Optional[str]:
        """Resolves the direct snapshot directory path to bypass HF Hub network/repo checks."""
        hub_name = "models--" + model_name.replace("/", "--")
        cache_path = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", hub_name))
        snapshots_dir = os.path.join(cache_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            try:
                snapshots = [
                    os.path.join(snapshots_dir, s)
                    for s in os.listdir(snapshots_dir)
                    if os.path.isdir(os.path.join(snapshots_dir, s))
                ]
                if snapshots:
                    return snapshots[0]
            except Exception:
                pass
        return None

    def _get_model(self):
        if not self._load_attempted:
            self._load_attempted = True
            
            # Prioritize BAAI/bge-small-en-v1.5 which is locally cached with dim=384
            candidates = []
            if self.model_name and self._is_model_cached(self.model_name):
                candidates.append(self.model_name)
            if self._is_model_cached("BAAI/bge-small-en-v1.5") and "BAAI/bge-small-en-v1.5" not in candidates:
                candidates.append("BAAI/bge-small-en-v1.5")

            loaded_model = None
            last_err = None
            for candidate in candidates:
                load_target = self._resolve_snapshot_dir(candidate) or candidate
                try:
                    from sentence_transformers import SentenceTransformer
                    if os.path.isdir(load_target):
                        loaded_model = SentenceTransformer(load_target)
                    else:
                        loaded_model = SentenceTransformer(load_target, local_files_only=True)
                    self.model_name = candidate
                    logger.info(f"Loaded neural embedding model from local cache: {candidate} ({load_target})")
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(f"Failed to load cached model {candidate}: {e}")

            if loaded_model is not None:
                self._model = loaded_model
            else:
                env = getattr(settings, "ENVIRONMENT", "").lower()
                if env in ["production", "staging"]:
                    raise RuntimeError(
                        f"CRITICAL: Neural embedding model could not be loaded from local cache. "
                        f"Checked candidates: {candidates}. Last error: {last_err}"
                    )
                logger.warning(
                    f"Neural embedding model not available in local cache. Deterministic fallback active for isolated unit tests only."
                )
                self._model = None
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate a normalized 384-dimensional vector embedding for a single text."""
        batch = self.embed_batch([text])
        return batch[0] if batch else [0.0] * self.dimension

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized 384-dimensional vector embeddings for a list of texts."""
        if not texts:
            return []

        model = self._get_model()
        if model is not None:
            try:
                embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                res = [e.tolist() for e in embeddings]
                # Strictly verify dimension is exactly 384
                for vec in res:
                    if len(vec) != self.dimension:
                        raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(vec)}")
                return res
            except Exception as e:
                logger.error(f"Error during neural model encoding: {e}")
                env = getattr(settings, "ENVIRONMENT", "").lower()
                if env in ["production", "staging"]:
                    raise

        # Deterministic offline fallback for unit tests running without models
        return [self._generate_deterministic_embedding(t) for t in texts]

    def _generate_deterministic_embedding(self, text: str) -> List[float]:
        """Generates a reproducible, unit-normalized dense vector of specified dimension."""
        if not text:
            return [0.0] * self.dimension

        # Generate seed from SHA-256
        seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes[:8], byteorder="big")
        rng = np.random.RandomState(seed % (2**32))

        # Random projection of text tokens
        vec = rng.randn(self.dimension).astype(np.float32)
        # Unit normalize for cosine distance
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


embedding_service = EmbeddingService()
