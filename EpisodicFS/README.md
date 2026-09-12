
# EpisodicFS: Privacy-Preserving On-Device Multimodal Retrieval for Snapdragon X

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qualcomm AI Hub](https://img.shields.io/badge/Qualcomm%20AI%20Hub-Ready-purple)
![LanceDB](https://img.shields.io/badge/LanceDB-Vector%20DB-green)
![Snapdragon X Compute](https://img.shields.io/badge/Snapdragon%20X%20Compute-Optimized-red)

## Executive Overview
EpisodicFS redefines on-device data retrieval by moving beyond traditional keyword searches and the privacy concerns of intrusive screen-recording solutions like Windows Recall. It builds a local, privacy-preserving **Spatio-Temporal Knowledge Graph** that connects your digital artifacts (images, documents, audio) not just by content, but by their *episodic context*—when and where they were created, and what other files co-occurred around those moments. This allows for more intuitive, context-aware, and multi-modal searches, optimized for the Qualcomm Hexagon NPU on Snapdragon X series devices.

Imagine asking, "Show me the bill from the restaurant where we discussed the server upgrade last month," and getting not just the invoice, but also the photo of the restaurant, the meeting notes, and a voice memo from that evening. EpisodicFS makes this possible by leveraging vector embeddings and temporal clustering.

## Architecture Diagram
```mermaid
graph TD
    A[Raw Files: Images, Docs, Audio] --> B{Feature Extraction: CLIP, MiniLM, Whisper}
    B --> C[Vector Embeddings (Hexagon NPU)]
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

## Snapdragon Acceleration Table
| Model (Sample) | Task                  | Runtime | Latency (ms) | Peak Memory (MB) | Hardware Target  |
|----------------|-----------------------|---------|--------------|------------------|------------------|
| MobileNetV2    | Image Classification  | QNN     | 12.5 (Mock)  | 35.2 (Mock)      | Hexagon NPU      |
| MiniLM-L6-v2   | Text Embedding        | ONNX    | N/A          | N/A              | Hexagon NPU      |
| Whisper-Base   | Speech-to-Text        | QNN     | N/A          | N/A              | Hexagon NPU      |

*Note: Benchmarks above are illustrative mock values. Actual performance will be determined via Qualcomm AI Hub profiling.*

## Quickstart Instructions

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Configure Qualcomm AI Hub (Optional, for NPU acceleration)
To use the actual Qualcomm AI Hub for model profiling and deployment, you need an API token. 
Visit [Qualcomm AI Hub](https://aihub.qualcomm.com/) to get your token.

In your Colab environment (or local environment if using `qai-hub` CLI):
- **Colab:** Click the '🔑' icon in the left sidebar, then 'Add a new secret'. Set the name to `QAI_TOKEN` and paste your API token as the value. Ensure 'Notebook access' is enabled.
- **Local CLI:** `qai-hub configure --api_token <YOUR_TOKEN>`

### 3. Ingest Your Data
EpisodicFS will scan a base directory, extract features, and build its knowledge graph.

```bash
python main.py ingest --base_dir ./my_vault --time_window_hours 2.0
```

This will process files in `./my_vault` and group them into episodes if their creation/modification times are within a 2-hour window.

### 4. Perform Semantic Queries

Query for documents, images, or audio using natural language.

```bash
python main.py query "meeting notes with Rahul about product launch" --top_k 5
```

To search without expanding to episodic context:

```bash
python main.py query "invoice for server upgrade" --no_episodic_context
```

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
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # Project overview and instructions
```