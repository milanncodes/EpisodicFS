
# EpisodicFS: Privacy-Preserving On-Device Multimodal Retrieval for Snapdragon X

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qualcomm AI Hub](https://img.shields.io/badge/Qualcomm%20AI%20Hub-Ready-purple)
![LanceDB](https://img.shields.io/badge/LanceDB-Vector%20DB-green)
![Snapdragon X Compute](https://img.shields.io/badge/Snapdragon%20X%20Compute-Optimized-red)

## Executive Overview
EpisodicFS is a local proof of concept for privacy-preserving semantic retrieval. It connects images, documents, and audio files by their temporal proximity, assigning shared episode IDs and linked file records. Queries return direct vector matches plus related files from the same episode.

The current implementation uses the 384-dimensional `all-MiniLM-L6-v2` model for text and PDF embeddings. Images and audio use deterministic schema-compatible fallback vectors; audio also records WAV metadata and duration. This keeps the local POC reproducible while leaving room for a future native multimodal or Qualcomm-accelerated extractor.

## Architecture Diagram
```mermaid
graph TD
    A[Raw Files: Images, Docs, Audio] --> B{Feature Extraction: MiniLM + Metadata}
    B --> C[384-D Vector Records]
    C --> D{Temporal & Co-occurrence Clustering}
    D --> E[Episodic Knowledge Graph]
    E --> F[LanceDB Vector Store]
    subgraph EpisodicFS Core
        B --&gt; C
        C --&gt; D
        D --&gt; E
        E --&gt; F
    end
    F --> G{Multi-Hop Search Engine}
    G --> H[Contextual Results]
```

## Snapdragon Profiling
| Model (Sample) | Task                  | Runtime | Latency (ms) | Peak Memory (MB) | Hardware Target  |
|----------------|-----------------------|---------|--------------|------------------|------------------|
| MobileCLIP     | Multimodal embedding  | QNN     | 4.2 (Mock)   | N/A              | Hexagon NPU      |
| Whisper-Base   | Speech-to-text        | QNN     | 12.8 (Mock)  | N/A              | Hexagon NPU      |

Run the optional profiler with `python main.py --profile-snapdragon`. Without credentials it prints a clearly labeled simulated diagnostic. With a configured SDK and token it attempts to submit a real AI Hub profile for the Snapdragon X Elite CRD.

## Quickstart Instructions

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Generate Test Fixtures
Create a fresh sample vault containing three PNG images, two text/Markdown documents, and one WAV file:

```bash
python generate_fixtures.py
```

The generated directories are `episodic_vault/images`, `episodic_vault/docs`, and `episodic_vault/audio`.

### 3. Configure Qualcomm AI Hub (Optional)
To attempt real Qualcomm AI Hub profiling, set a token from [Qualcomm AI Hub](https://aihub.qualcomm.com/):

```bash
export QAI_HUB_API_TOKEN="<YOUR_TOKEN>"
python main.py --profile-snapdragon
```

`QAI_TOKEN` is also accepted as a compatibility fallback. If the SDK or token is unavailable, the profiler remains local-only and prints simulated diagnostics.

### 4. Ingest Your Data
EpisodicFS will scan a base directory, extract features, and build its knowledge graph.

```bash
python main.py ingest --base_dir ./my_vault --time_window_hours 2.0
```

This processes supported files in `./my_vault/images`, `./my_vault/docs`, and `./my_vault/audio`, then groups records by modification time when adjacent files are within the selected time window.

### 5. Perform Semantic Queries

Query for documents, images, or audio using natural language.

```bash
python main.py query "meeting notes with Rahul about product launch" --top_k 5
```

To search without expanding to episodic context:

```bash
python main.py query "invoice for server upgrade" --no_episodic_context
```

If the table does not exist, the query command explains that ingestion must be run first.

## Project Structure

```
episodicfs_repo/
├── src/
│   ├── __init__.py
│   ├── features.py      # Embedding extraction for images, text, audio
│   ├── graph.py         # Episodic clustering and linking logic
│   ├── db.py            # LanceDB connection and ingestion
│   └── search.py        # Multi-hop query engine
├── main.py              # CLI entry point
├── generate_fixtures.py # Local sample-vault generator
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # Project overview and instructions
```