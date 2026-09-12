
import os
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

is_clip_model = False # Flag to indicate if a CLIP-based model was loaded
VECTOR_DIMENSION = 384 # Default dimension for all-MiniLM-L6-v2, updated if CLIP loads

try:
    # Attempt to load a multimodal model
    model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')
    is_clip_model = True
    VECTOR_DIMENSION = 512 # CLIP-ViT-B-32 typically outputs 512-d embeddings
    print(f"Loaded multimodal embedding model: clip-ViT-B-32 with dimension {VECTOR_DIMENSION}")
except Exception as e:
    print(f"Could not load clip-ViT-B-32 directly: {e}")
    print("Falling back to text-only model and will use placeholder for image embedding.")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    VECTOR_DIMENSION = model.get_sentence_embedding_dimension() # Should be 384
    print(f"Loaded text embedding model: all-MiniLM-L6-v2 with dimension {VECTOR_DIMENSION}")

    # Define get_image_embedding here, it will be used if not a CLIP model
    def get_image_embedding(image_path):
        print(f"Warning: Image embedding is a placeholder for {image_path}. No actual image embedding generated.")
        return np.random.rand(VECTOR_DIMENSION) # Dummy vector of correct dimension

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
            if is_clip_model: # Use the flag to check if a CLIP model was loaded
                img = Image.open(file_path).convert('RGB')
                embedding = model.encode(img, convert_to_numpy=True)
            else:
                embedding = get_image_embedding(file_path) # Use placeholder for image
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            embedding = np.random.rand(VECTOR_DIMENSION) # Fallback to dummy embedding of correct dim

    elif file_path.endswith(('.txt', '.md')):
        file_type = 'text'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            embedding = model.encode(text_content, convert_to_numpy=True)
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            embedding = np.random.rand(VECTOR_DIMENSION) # Fallback to dummy embedding of correct dim

    elif file_path.endswith(('.pdf')):
        file_type = 'document'
        try:
            reader = pypdf.PdfReader(file_path)
            text_content = "
".join([page.extract_text() for page in reader.pages if page.extract_text()])
            embedding = model.encode(text_content, convert_to_numpy=True)
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            embedding = np.random.rand(VECTOR_DIMENSION) # Fallback to dummy embedding of correct dim

    elif file_path.endswith(('.wav', '.mp3')):
        file_type = 'audio'
        # For audio, Whisper-Base would be used. Placeholder for now.
        print(f"Warning: Audio processing for {file_path} is a placeholder. No actual audio embedding generated.")
        text_content = f"Audio file: {filename}"
        embedding = np.random.rand(VECTOR_DIMENSION) # Dummy embedding of correct dim

    else:
        file_type = 'unknown'
        print(f"Warning: Unknown file type for {file_path}. No embedding generated.")
        embedding = np.random.rand(VECTOR_DIMENSION) # Dummy embedding of correct dim

    if embedding is not None:
        # print(f"DEBUG: {file_path} generated embedding of dimension: {len(embedding)}") # Keep for debugging
        pass

    return {
        'id': str(hash(file_path)), # Simple unique ID for now
        'file_path': file_path,
        'filename': filename,
        'file_type': file_type,
        'timestamp': timestamp,
        'file_size': file_size,
        'text_content': text_content,
        'vector': embedding.tolist() if embedding is not None else None # Convert numpy array to list for storage
    }