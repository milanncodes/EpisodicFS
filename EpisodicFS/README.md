# EpisodicFS

## On-Device Memory for Snapdragon X

EpisodicFS is an on-device, zero-cloud semantic and episodic file-system memory layer powered by the Snapdragon X Hexagon NPU. It turns local files into searchable memories, then reconstructs the surrounding work episode: the notes, code, invoices, transcripts, and assets that were created or modified together.

The system combines keyword retrieval, ONNX embeddings, and a temporal-semantic knowledge graph. It is designed for private, offline-first workflows on Snapdragon X Windows ARM64 PCs, including HP Omnibook systems.

## Why On-Device Snapdragon

### Strict privacy

Documents, source code, meeting transcripts, invoices, and embeddings remain on the PC. EpisodicFS sends no enterprise or personal documents to a cloud retrieval service, which keeps sensitive work material inside the device boundary.

### Continuous, low-power indexing

The Hexagon HTP is built for efficient inference close to the data. Background embedding and episodic linking can run locally without continuously waking a high-power cloud connection or moving files over the network.

### Instant offline retrieval

Search remains available when disconnected from the internet. Local lexical indexes, vector embeddings, and the episodic graph provide fast retrieval without network round trips or the battery cost of uploading and downloading document content.

## Architecture Flowchart

```mermaid
graph LR
    A[User Query] --> B[Hybrid Retrieval<br/>FTS5 + SnapdragonEmbedder]
    B --> C[ONNX Runtime<br/>QNN HTP Provider]
    C --> D[Episodic Graph Expansion<br/>Temporal + Semantic]
    D --> E[Result Ranking]
```

The dense path uses `SnapdragonEmbedder` through ONNX Runtime. When `QNNExecutionProvider` and the Hexagon HTP backend are available, inference is targeted to the NPU; otherwise, the profiler and embedder report an explicit CPU fallback. The lexical path ranks local file text and metadata, and Reciprocal Rank Fusion combines both rankings before episodic expansion.

## Benchmark & Telemetry

The profiler reports p50, p95, average latency, embeddings per second, active provider, backend path, and memory delta. The following values are representative presentation targets, not a substitute for a device run:

| Workload | Hexagon HTP / QNN | CPU fallback | Battery and responsiveness profile |
|---|---:|---:|---|
| Text embedding batch | 7.8 ms | 45.2 ms | HTP keeps background indexing responsive and reduces sustained CPU work |
| Episodic query embedding | 8.4 ms | 48.6 ms | Local HTP inference avoids network transfer and keeps offline search interactive |
| 10-document indexing batch | 78 embeddings/sec | 17 embeddings/sec | HTP is better suited to continuous, low-power indexing |

Run a device-specific A/B measurement with:

```bash
python main.py profile --model-path models/embedding_model.onnx
```

The report compares QNN/Hexagon HTP with `CPUExecutionProvider`, including speedup factor and latency delta. If the QNN backend DLL or provider is unavailable, the result is clearly marked as CPU fallback rather than presenting simulated NPU measurements.

## Quickstart & Verification

1. Install the dependencies in a Python 3.10–3.12 environment. On Windows ARM64, use the matching `onnxruntime-qnn` wheel for the Snapdragon QNN provider.

   ```bash
   pip install -r requirements.txt
   ```

2. Generate the synthetic work history, index it into LanceDB, build the temporal-semantic graph, and run the three built-in episodic queries:

   ```bash
   python main.py demo
   ```

3. Verify the output contains hybrid RRF scores, episodic context explanations, and a hardware badge such as `[NPU: Hexagon HTP - 7.8ms]` or `[CPU Fallback - 45.2ms]`. To query the generated vault again, use:

   ```bash
   python main.py search "event pipeline retry queue latency"
   ```

The demo uses a temporary fixture vault by default, so it does not scan or modify personal files. To index an existing local vault, use `python main.py ingest --base-dir <path>`.
