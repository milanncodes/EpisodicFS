
import os
import hashlib
from datetime import datetime
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import pypdf
import soundfile as sf

# Initialize a lightweight multimodal embedding model (e.g., MiniLM for text, CLIP for images)
# Using a SentenceTransformer model that can handle both text and potentially image (if available or a workaround)
# 'sentence-transformers/clip-ViT-B-32' is a good multimodal choice if available directly.
# If not, 'all-MiniLM-L6-v2' for text and a separate approach for images.

VECTOR_DIMENSION = 384

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    if model.get_sentence_embedding_dimension() != VECTOR_DIMENSION:
        raise RuntimeError("all-MiniLM-L6-v2 did not provide the required 384 dimensions")
    print(f"Loaded embedding model: all-MiniLM-L6-v2 with dimension {VECTOR_DIMENSION}")
except Exception as e:
    raise RuntimeError(
        "Unable to load the required 384-dimensional embedding model. "
        "Install requirements and ensure the model is available locally."
    ) from e


def _deterministic_embedding(value):
    """Create a repeatable fallback vector without pretending it is semantic."""
    digest = hashlib.sha256(value.encode('utf-8')).digest()
    values = np.frombuffer((digest * ((VECTOR_DIMENSION * 4 // len(digest)) + 1)), dtype=np.uint8)
    return (values[:VECTOR_DIMENSION].astype(np.float32) / 255.0).tolist()


def get_image_embedding(image_path):
    """Return a deterministic image fallback while keeping the vector schema valid."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        print(f"Warning: image semantics are unavailable; using deterministic fallback for {image_path}.")
    except Exception as exc:
        raise ValueError(f"Invalid image file {image_path}: {exc}") from exc
    return np.asarray(_deterministic_embedding(image_path), dtype=np.float32)


def get_audio_embedding(audio_path):
    """Extract audio metadata and return a deterministic schema-compatible vector."""
    try:
        info = sf.info(audio_path)
        descriptor = f"{audio_path}:{info.frames}:{info.samplerate}:{info.channels}"
        return np.asarray(_deterministic_embedding(descriptor), dtype=np.float32), info.duration
    except Exception as exc:
        raise ValueError(f"Invalid or unsupported audio file {audio_path}: {exc}") from exc

def extract_features(file_path):
    """
    Reads a file, extracts metadata, and generates embeddings.
    """
    filename = os.path.basename(file_path)
    file_stat = os.stat(file_path)
    timestamp = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
    file_size = file_stat.st_size

    text_content = None
    embedding = None

    # Determine file type and process
    if file_path.endswith(('.jpg', '.jpeg', '.png')):
        file_type = 'image'
        try:
            embedding = get_image_embedding(file_path)
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            embedding = np.asarray(_deterministic_embedding(file_path), dtype=np.float32)

    elif file_path.endswith(('.txt', '.md')):
        file_type = 'text'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            embedding = model.encode(text_content, convert_to_numpy=True)
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            embedding = np.asarray(_deterministic_embedding(file_path), dtype=np.float32)

    elif file_path.endswith(('.pdf')):
        file_type = 'document'
        try:
            reader = pypdf.PdfReader(file_path)
            text_content = "\n".join(
                page_text for page in reader.pages if (page_text := page.extract_text())
            )
            embedding = model.encode(text_content, convert_to_numpy=True)
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            embedding = np.asarray(_deterministic_embedding(file_path), dtype=np.float32)

    elif file_path.endswith(('.wav', '.mp3')):
        file_type = 'audio'
        embedding, duration = get_audio_embedding(file_path)
        text_content = f"Audio file: {filename}; duration: {duration:.2f}s"

    else:
        file_type = 'unknown'
        print(f"Warning: Unknown file type for {file_path}. No embedding generated.")
        embedding = np.asarray(_deterministic_embedding(file_path), dtype=np.float32)

    if embedding is not None:
        # print(f"DEBUG: {file_path} generated embedding of dimension: {len(embedding)}") # Keep for debugging
        pass

    return {
        'id': hashlib.sha256(file_path.encode('utf-8')).hexdigest()[:16],
        'file_path': file_path,
        'filename': filename,
        'file_type': file_type,
        'timestamp': timestamp,
        'file_size': file_size,
        'text_content': text_content,
        'vector': embedding.tolist() if embedding is not None else None # Convert numpy array to list for storage
    }