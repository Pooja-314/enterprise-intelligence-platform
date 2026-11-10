import hashlib
import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np

from app.config import settings

# Force single-threading across CPU math libraries to stay strictly within 512MB RAM limit on Render
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)


def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in megabytes."""
    try:
        import psutil
        p = psutil.Process()
        return round(p.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0


def release_system_memory() -> None:
    """Forces Python garbage collection and returns freed heap pages back to the Linux OS/cgroup."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class ONNXEmbeddingEngine:
    """Ultra-lightweight neural embedding engine powered by pure ONNX Runtime and Tokenizers.

    Eliminates PyTorch and SentenceTransformers from the production inference path,
    consuming ~65MB total memory while producing strictly identical 384-dimensional
    unit-normalized dense vectors for pgvector compatibility.
    """

    def __init__(self, model_dir: str, dimension: int = 384):
        from tokenizers import Tokenizer
        import onnxruntime as ort

        self.dimension = dimension
        self.model_dir = model_dir

        tok_path = os.path.join(model_dir, "tokenizer.json")
        onnx_path = os.path.join(model_dir, "onnx", "model.onnx")
        if not os.path.isfile(onnx_path):
            onnx_path = os.path.join(model_dir, "model.onnx")

        if not os.path.isfile(tok_path):
            raise FileNotFoundError(f"Missing tokenizer.json in: {model_dir}")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"Missing model.onnx in: {model_dir}")

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(length=512)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        # Disable CPU memory arena: prevents ONNX Runtime from hoarding pre-allocated memory
        opts.enable_cpu_mem_arena = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        batch_size: int = 8,
        **kwargs
    ) -> np.ndarray:
        """Encodes texts into dense embeddings using BERT CLS token pooling."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            sub_texts = texts[i : i + batch_size]
            encoded = self.tokenizer.encode_batch(sub_texts)
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )

            # CLS token pooling (BAAI/bge-small-en-v1.5 uses first token [CLS] at index 0)
            cls_embeddings = outputs[0][:, 0].astype(np.float32)

            if normalize_embeddings:
                norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                cls_embeddings = cls_embeddings / norms

            all_embs.append(cls_embeddings)

        return np.vstack(all_embs) if all_embs else np.empty((0, self.dimension), dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return self.dimension


class EmbeddingService:
    """Service for generating dense vector embeddings with ultra-low memory footprint.
    
    Loads locally cached 384-dimensional neural models (specifically BAAI/bge-small-en-v1.5)
    using pure ONNX Runtime in production to operate comfortably within Render Free's 512MB RAM.
    """

    def __init__(self, model_name: Optional[str] = None, dimension: Optional[int] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self._model = None
        self._load_attempted = False
        self._is_quantized = False
        self._precision = "fp32"
        self._engine = "onnxruntime"
        self._onnx_loader = None         # Hook for custom ONNX loader injection in tests
        self._transformer_loader = None  # Hook for custom SentenceTransformer loader injection in tests

    def _load_onnx_engine(self, target_dir: str) -> Any:
        """Internal helper to instantiate ONNXEmbeddingEngine, allowing test isolation."""
        if self._onnx_loader is not None:
            return self._onnx_loader(target_dir)
        return ONNXEmbeddingEngine(target_dir, dimension=self.dimension)

    def _load_sentence_transformer(self, target: str, **kwargs) -> Any:
        """Internal helper to instantiate SentenceTransformer (testing and local fallback only)."""
        if self._transformer_loader is not None:
            return self._transformer_loader(target, **kwargs)
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        kwargs.setdefault("device", "cpu")
        try:
            import torch
            kwargs.setdefault("model_kwargs", {"dtype": torch.bfloat16, "low_cpu_mem_usage": True})
        except Exception:
            kwargs.setdefault("model_kwargs", {"low_cpu_mem_usage": True})
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
        snapshot_dir = self._resolve_snapshot_dir(model_name)
        if snapshot_dir and os.path.isdir(snapshot_dir):
            onnx_path = os.path.join(snapshot_dir, "onnx", "model.onnx")
            if not os.path.isfile(onnx_path):
                onnx_path = os.path.join(snapshot_dir, "model.onnx")
            sf_path = os.path.join(snapshot_dir, "model.safetensors")
            return os.path.isfile(onnx_path) or os.path.isfile(sf_path)
        return False

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
        if self._model is not None:
            return self._model

        if not self._load_attempted:
            self._load_attempted = True
            
            loaded_model = None
            last_err = None
            detected_engine = "onnxruntime"

            candidates = [self.model_name]
            if "BAAI/bge-small-en-v1.5" not in candidates:
                candidates.append("BAAI/bge-small-en-v1.5")

            # Phase 1: Try pure ONNX Runtime engine from local Hugging Face cache (ultra-low memory)
            for candidate in candidates:
                snapshot_dir = self._resolve_snapshot_dir(candidate)
                if snapshot_dir:
                    onnx_path = os.path.join(snapshot_dir, "onnx", "model.onnx")
                    if not os.path.isfile(onnx_path):
                        onnx_path = os.path.join(snapshot_dir, "model.onnx")
                    if os.path.isfile(onnx_path) or self._onnx_loader is not None:
                        try:
                            loaded_model = self._load_onnx_engine(snapshot_dir)
                            detected_engine = "onnxruntime"
                            self.model_name = candidate
                            logger.info(f"Loaded ultra-lightweight ONNX neural embedding model from: {candidate}")
                            break
                        except Exception as e:
                            last_err = e
                            logger.debug(f"Local ONNX load for {candidate} failed: {e}")

            # Phase 2: If ONNX model not cached locally, attempt online download of ONNX files from Hugging Face Hub
            if loaded_model is None:
                for candidate in candidates:
                    try:
                        logger.info(f"Downloading ONNX neural model files for '{candidate}' from Hugging Face Hub...")
                        from huggingface_hub import hf_hub_download
                        for f in ["onnx/model.onnx", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "config.json"]:
                            hf_hub_download(candidate, f)
                        snapshot_dir = self._resolve_snapshot_dir(candidate)
                        if snapshot_dir:
                            loaded_model = self._load_onnx_engine(snapshot_dir)
                            detected_engine = "onnxruntime"
                            self.model_name = candidate
                            logger.info(f"Successfully downloaded and initialized ONNX neural model: {candidate}")
                            break
                    except Exception as e:
                        last_err = e
                        logger.debug(f"ONNX download attempt for {candidate} failed: {e}")

            # Phase 3: Non-production fallback to SentenceTransformer (only if test injected loader or non-production environment)
            env = getattr(settings, "ENVIRONMENT", "").lower()
            is_prod = env in ["production", "staging"]

            if loaded_model is None and (not is_prod or self._transformer_loader is not None):
                for candidate in candidates:
                    snapshot_dir = self._resolve_snapshot_dir(candidate)
                    target = snapshot_dir if (snapshot_dir and os.path.isdir(snapshot_dir)) else candidate
                    try:
                        loaded_model = self._load_sentence_transformer(target, local_files_only=True)
                        detected_engine = "sentence_transformers"
                        self.model_name = candidate
                        logger.info(f"Loaded neural embedding model via SentenceTransformer fallback: {candidate}")
                        break
                    except Exception as e:
                        last_err = e
                        logger.debug(f"SentenceTransformer fallback for {candidate} failed: {e}")

            # Phase 4: Validate dimension, set precision state, and release memory
            if loaded_model is not None:
                dim_fn = getattr(loaded_model, "get_embedding_dimension", None) or getattr(
                    loaded_model, "get_sentence_embedding_dimension", lambda: None
                )
                actual_dim = dim_fn()
                if actual_dim is not None and actual_dim != self.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.dimension}, got {actual_dim} from {self.model_name}"
                    )
                
                self._model = loaded_model
                self._engine = detected_engine
                self._is_quantized = False
                self._precision = "fp32"
                logger.info(
                    f"Embedding model '{self.model_name}' active with engine '{self._engine}'. "
                    f"Current RSS: {get_process_memory_mb()}MB"
                )
                release_system_memory()
            else:
                if is_prod:
                    raise RuntimeError(
                        f"CRITICAL: Production ONNX neural embedding model could not be loaded for '{self.model_name}'. "
                        f"Checked candidates: {candidates}. Last error: {last_err}. "
                        f"PyTorch fallback is strictly disabled in production to protect against 512MB OOM crash."
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
            dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension", lambda: self.dimension
            )
            actual_dim = dim_fn()

        status = "loaded" if is_loaded else ("cached" if is_cached else "ready_for_download")
        return {
            "model": self.model_name,
            "engine": getattr(self, "_engine", "onnxruntime"),
            "configured_dimension": self.dimension,
            "actual_dimension": actual_dim or self.dimension,
            "is_loaded": is_loaded,
            "is_cached": is_cached,
            "is_quantized": getattr(self, "_is_quantized", False),
            "precision": getattr(self, "_precision", "fp32"),
            "process_rss_mb": get_process_memory_mb(),
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
                embeddings = model.encode(
                    texts,
                    batch_size=min(8, len(texts)),
                    normalize_embeddings=True,
                )
                # Strictly ensure 32-bit floating point precision and exact unit L2 normalization
                arr = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr = arr / norms
                res = arr.tolist()
                del embeddings, arr

                # Strictly verify dimension is exactly 384
                for vec in res:
                    if len(vec) != self.dimension:
                        raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(vec)}")

                if len(texts) > 1:
                    release_system_memory()
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
