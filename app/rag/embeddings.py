import hashlib
import logging
import os
from typing import Any, Dict, List, Optional
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
        self._transformer_loader = None  # Hook for custom loader injection (e.g. testing)

    def _load_sentence_transformer(self, target: str, **kwargs) -> Any:
        """Internal helper to instantiate SentenceTransformer, allowing test isolation."""
        if self._transformer_loader is not None:
            return self._transformer_loader(target, **kwargs)
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(target, **kwargs)

    def _get_hf_hub_dir(self) -> str:
        """Determines the Hugging Face hub cache directory respecting HF_HOME."""
        hf_home = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
        if hf_home:
            return hf_home if "hub" in hf_home else os.path.join(hf_home, "hub")
        return os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub"))

    def _is_model_cached(self, model_name: str) -> bool:
        """Checks if model files are cached locally to avoid slow network timeouts."""
        hub_name = "models--" + model_name.replace("/", "--")
        cache_path = os.path.join(self._get_hf_hub_dir(), hub_name)
        return os.path.isdir(cache_path)

    def _resolve_snapshot_dir(self, model_name: str) -> Optional[str]:
        """Resolves the direct snapshot directory path to bypass HF Hub network/repo checks."""
        hub_name = "models--" + model_name.replace("/", "--")
        cache_path = os.path.join(self._get_hf_hub_dir(), hub_name)
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
            
            loaded_model = None
            last_err = None

            # Primary candidate is the configured model, followed by default BAAI/bge-small-en-v1.5
            candidates = [self.model_name]
            if "BAAI/bge-small-en-v1.5" not in candidates:
                candidates.append("BAAI/bge-small-en-v1.5")

            # Phase 1: Try loading from local Hugging Face cache (offline, fast)
            for candidate in candidates:
                snapshot_dir = self._resolve_snapshot_dir(candidate)
                target = snapshot_dir if (snapshot_dir and os.path.isdir(snapshot_dir)) else candidate
                try:
                    loaded_model = self._load_sentence_transformer(target, local_files_only=True)
                    self.model_name = candidate
                    logger.info(f"Loaded neural embedding model from local cache: {candidate}")
                    break
                except Exception as e:
                    last_err = e
                    logger.debug(f"Local cache load for {candidate} failed: {e}")

            # Phase 2: If not found in local cache, download from Hugging Face Hub
            if loaded_model is None:
                for candidate in candidates:
                    try:
                        logger.info(f"Downloading neural embedding model from Hugging Face Hub: {candidate}...")
                        loaded_model = self._load_sentence_transformer(candidate)
                        self.model_name = candidate
                        logger.info(f"Successfully downloaded and loaded neural embedding model: {candidate}")
                        break
                    except Exception as e:
                        last_err = e
                        logger.warning(f"Failed to download neural embedding model {candidate} from Hub: {e}")

            # Phase 3: Validate dimension and handle environment
            if loaded_model is not None:
                actual_dim = getattr(loaded_model, "get_sentence_embedding_dimension", lambda: None)()
                if actual_dim is not None and actual_dim != self.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.dimension}, got {actual_dim} from {self.model_name}"
                    )
                self._model = loaded_model
            else:
                env = getattr(settings, "ENVIRONMENT", "").lower()
                if env in ["production", "staging"]:
                    raise RuntimeError(
                        f"CRITICAL: Neural embedding model could not be loaded or downloaded for '{self.model_name}'. "
                        f"Checked candidates: {candidates}. Last error: {last_err}"
                    )
                logger.warning(
                    f"Neural embedding model not available (error: {last_err}). Deterministic fallback active for isolated tests only."
                )
                self._model = None
        return self._model

    def get_diagnostic(self) -> dict:
        """Returns diagnostic health and initialization status of the embedding service."""
        is_cached = self._is_model_cached(self.model_name)
        is_loaded = self._model is not None
        actual_dim = None
        if self._model is not None:
            actual_dim = getattr(self._model, "get_sentence_embedding_dimension", lambda: self.dimension)()

        status = "loaded" if is_loaded else ("cached" if is_cached else "ready_for_download")
        return {
            "model": self.model_name,
            "configured_dimension": self.dimension,
            "actual_dimension": actual_dim or self.dimension,
            "is_loaded": is_loaded,
            "is_cached": is_cached,
            "cache_dir": self._get_hf_hub_dir(),
            "status": status,
        }

    def warmup(self) -> bool:
        """Warms up the embedding model in a non-blocking background task."""
        try:
            self._get_model()
            logger.info(f"Embedding model '{self.model_name}' successfully warmed up.")
            return True
        except Exception as e:
            logger.warning(f"Embedding model warmup deferred: {e}")
            return False

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
