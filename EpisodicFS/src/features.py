import hashlib
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pypdf
import soundfile as sf
from PIL import Image

from .qualcomm_profiler import CPU_PROVIDER, QNN_PROVIDER, QualcommProfiler


VECTOR_DIMENSION = 384
DEFAULT_MODEL_PATH = "models/embedding_model.onnx"
DEFAULT_TOKENIZER_PATH = "models/embedding_model"
QNN_PROVIDER_OPTIONS = {
    "backend_path": "QnnHtp.dll",
    "htp_performance_mode": "burst",
    "htp_graph_finalization_optimization_mode": "3",
}


def _warn(message: str) -> None:
    print(f"Warning: {message}")


class SnapdragonEmbedder:
    """Run transformer embeddings through Qualcomm QNN HTP when available."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        tokenizer_path: Optional[str] = DEFAULT_TOKENIZER_PATH,
        profiler: Optional[QualcommProfiler] = None,
        max_length: int = 512,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        self.model_path = model_path
        self.max_length = max_length
        self.profiler = profiler or QualcommProfiler()
        self.active_provider = CPU_PROVIDER
        self._runtime = self._load_runtime()
        self._tokenizer = self._load_tokenizer(tokenizer_path or model_path)
        self._session = self._create_session()

    @staticmethod
    def _load_runtime() -> Any:
        try:
            import onnxruntime as runtime
        except ImportError as exc:
            _warn("ONNX Runtime is unavailable; Snapdragon embedding execution cannot start.")
            raise RuntimeError(
                "ONNX Runtime is required for Snapdragon embeddings. "
                "Install onnxruntime and the Qualcomm QNN execution provider."
            ) from exc
        return runtime

    @staticmethod
    def _load_tokenizer(tokenizer_path: str) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            _warn("Transformers is unavailable; local embedding tokenization cannot start.")
            raise RuntimeError(
                "Transformers is required to tokenize Snapdragon embeddings."
            ) from exc
        try:
            return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception as exc:
            _warn(f"Unable to load the local tokenizer at {tokenizer_path}: {exc}")
            raise RuntimeError(
                f"Unable to load the local tokenizer at {tokenizer_path}: {exc}"
            ) from exc

    def _create_session(self) -> Any:
        try:
            available = set(self._runtime.get_available_providers())
        except Exception as exc:
            _warn(f"Could not inspect ONNX Runtime execution providers: {exc}")
            available = set()
        if QNN_PROVIDER in available:
            try:
                session = self._runtime.InferenceSession(
                    self.model_path,
                    providers=[(QNN_PROVIDER, QNN_PROVIDER_OPTIONS), CPU_PROVIDER],
                )
                self.active_provider = QNN_PROVIDER
                return session
            except Exception as exc:
                _warn(
                    "Qualcomm QNN/Hexagon HTP session creation failed; "
                    f"falling back to CPUExecutionProvider ({exc})."
                )
                print(
                    "Warning: Qualcomm QNN/Hexagon HTP unavailable; "
                    f"falling back to CPUExecutionProvider ({exc})."
                )
        else:
            print(
                "Warning: QNNExecutionProvider or the HTP driver is unavailable; "
                "falling back to CPUExecutionProvider."
            )

        try:
            return self._runtime.InferenceSession(
                self.model_path,
                providers=[CPU_PROVIDER],
            )
        except Exception as exc:
            _warn(f"CPUExecutionProvider session creation failed: {exc}")
            raise RuntimeError(
                f"Unable to create an ONNX Runtime CPU session: {exc}"
            ) from exc

    def _tokenize(self, texts: List[str]) -> Mapping[str, np.ndarray]:
        try:
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
        except Exception as exc:
            _warn(f"Embedding tokenization failed: {exc}")
            raise
        return {name: np.asarray(value) for name, value in encoded.items()}

    def _session_inputs(self, tokenized: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inputs: Dict[str, np.ndarray] = {}
        for input_meta in self._session.get_inputs():
            name = input_meta.name
            if name in tokenized:
                inputs[name] = tokenized[name]
            elif name == "token_type_ids":
                inputs[name] = np.zeros_like(tokenized["input_ids"])
            else:
                raise ValueError(f"Tokenizer did not produce required ONNX input: {name}")
        return inputs

    @staticmethod
    def _mean_pool(outputs: Any, attention_mask: np.ndarray) -> np.ndarray:
        hidden_states = np.asarray(outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
        if hidden_states.ndim == 2:
            return hidden_states.astype(np.float32, copy=False)
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected transformer output with 2 or 3 dimensions, got {hidden_states.shape}"
            )
        mask = attention_mask[:, :, None].astype(np.float32)
        pooled = (hidden_states * mask).sum(axis=1)
        return (pooled / np.maximum(mask.sum(axis=1), 1e-9)).astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Return one mean-pooled embedding per input text."""
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list of strings")
        if not texts:
            return np.empty((0, VECTOR_DIMENSION), dtype=np.float32)

        try:
            tokenized = self._tokenize(texts)
            inputs = self._session_inputs(tokenized)
            started = time.perf_counter()
            outputs = self._session.run(None, inputs)
            latency_ms = (time.perf_counter() - started) * 1000.0
            embeddings = self._mean_pool(outputs, inputs["attention_mask"])
        except Exception as exc:
            _warn(f"On-device embedding inference failed: {exc}")
            raise
        self.profiler.record_latency(latency_ms, work_units=len(texts))
        return embeddings

    def embed_text(self, text: str) -> np.ndarray:
        """Return a single mean-pooled embedding."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self.embed_batch([text])[0]

    def encode(self, texts: Any, convert_to_numpy: bool = True, **_: Any) -> np.ndarray:
        """SentenceTransformer-compatible alias used by the existing search path."""
        embeddings = self.embed_text(texts) if isinstance(texts, str) else self.embed_batch(list(texts))
        return embeddings if convert_to_numpy else embeddings.tolist()


class _LazyDefaultEmbedder:
    def __init__(self) -> None:
        self._instance: Optional[SnapdragonEmbedder] = None

    def _get(self) -> SnapdragonEmbedder:
        if self._instance is None:
            self._instance = SnapdragonEmbedder()
        return self._instance

    def encode(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._get().encode(*args, **kwargs)


model = _LazyDefaultEmbedder()


def _deterministic_embedding(value: str) -> np.ndarray:
    """Create a repeatable fallback vector without pretending it is semantic."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    values = np.frombuffer((digest * ((VECTOR_DIMENSION * 4 // len(digest)) + 1)), dtype=np.uint8)
    return values[:VECTOR_DIMENSION].astype(np.float32) / 255.0


def get_image_embedding(image_path: str) -> np.ndarray:
    try:
        with Image.open(image_path) as image:
            image.verify()
        print(f"Warning: image semantics are unavailable; using deterministic fallback for {image_path}.")
    except Exception as exc:
        _warn(f"Image validation failed for {image_path}: {exc}")
        raise ValueError(f"Invalid image file {image_path}: {exc}") from exc
    return _deterministic_embedding(image_path)


def get_audio_embedding(audio_path: str):
    try:
        info = sf.info(audio_path)
        descriptor = f"{audio_path}:{info.frames}:{info.samplerate}:{info.channels}"
        return _deterministic_embedding(descriptor), info.duration
    except Exception as exc:
        _warn(f"Audio metadata extraction failed for {audio_path}: {exc}")
        raise ValueError(f"Invalid or unsupported audio file {audio_path}: {exc}") from exc


def extract_features(file_path: str):
    filename = os.path.basename(file_path)
    file_stat = os.stat(file_path)
    timestamp = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
    file_size = file_stat.st_size
    text_content = None

    if file_path.endswith((".jpg", ".jpeg", ".png")):
        file_type = "image"
        try:
            embedding = get_image_embedding(file_path)
        except Exception as exc:
            print(f"Error processing image {file_path}: {exc}")
            embedding = _deterministic_embedding(file_path)
    elif file_path.endswith((".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml", ".sql")):
        file_type = "text"
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text_content = file.read()
            embedding = model.encode(text_content, convert_to_numpy=True)
        except Exception as exc:
            print(f"Error processing text file {file_path}: {exc}")
            embedding = _deterministic_embedding(file_path)
    elif file_path.endswith(".pdf"):
        file_type = "document"
        try:
            reader = pypdf.PdfReader(file_path)
            text_content = "\n".join(
                page_text for page in reader.pages if (page_text := page.extract_text())
            )
            embedding = model.encode(text_content, convert_to_numpy=True)
        except Exception as exc:
            print(f"Error processing PDF {file_path}: {exc}")
            embedding = _deterministic_embedding(file_path)
    elif file_path.endswith((".wav", ".mp3")):
        file_type = "audio"
        embedding, duration = get_audio_embedding(file_path)
        text_content = f"Audio file: {filename}; duration: {duration:.2f}s"
    else:
        file_type = "unknown"
        print(f"Warning: Unknown file type for {file_path}. No embedding generated.")
        embedding = _deterministic_embedding(file_path)

    return {
        "id": hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16],
        "file_path": file_path,
        "filename": filename,
        "file_type": file_type,
        "timestamp": timestamp,
        "file_size": file_size,
        "text_content": text_content,
        "vector": np.asarray(embedding, dtype=np.float32).tolist(),
    }
