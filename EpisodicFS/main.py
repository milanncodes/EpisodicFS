"""Unified command-line interface for EpisodicFS."""

import argparse
import os
from typing import Any, Dict, Optional


DEFAULT_TIME_WINDOW_SECONDS = 30 * 60


def _connect(base_dir: str):
    try:
        import lancedb
    except ImportError as exc:
        raise RuntimeError("LanceDB is required. Install dependencies with: pip install -r requirements.txt") from exc
    return lancedb.connect(os.path.join(base_dir, "db"))


def _load_graph(base_dir: str):
    from src.db import load_episodic_graph

    try:
        return load_episodic_graph(base_dir)
    except RuntimeError:
        return None


def _hardware_badge(telemetry: Dict[str, Any]) -> str:
    if telemetry.get("is_npu_active"):
        latency = telemetry.get("average_latency_ms")
        return f"[NPU: Hexagon HTP - {latency:.1f}ms]" if latency is not None else "[NPU: Hexagon HTP]"
    latency = telemetry.get("average_latency_ms")
    return f"[CPU Fallback - {latency:.1f}ms]" if latency is not None else "[CPU Fallback]"


def _relationship_reason(result: Dict[str, Any]) -> str:
    source = result.get("match_source", "unknown")
    record = result.get("record", result)
    relationships = record.get("relationship", [])
    if isinstance(relationships, str):
        relationships = [relationships]
    if "TEMPORAL_PROXIMITY" in relationships:
        return "Modified during the same 30m window as the seed file."
    if "SEMANTIC_SIMILARITY" in relationships:
        return "Semantically similar to the seed file's embedding."
    if "DIRECTORY_SIBLING" in relationships:
        return "Stored beside a file in the same working directory."
    if "episodic" in source:
        return "Connected through the episodic graph."
    if "vector" in source and "keyword" in source:
        return "Matched by both dense vector and keyword ranking."
    if "vector" in source:
        return "Matched by dense Qualcomm embedding search."
    if "keyword" in source:
        return "Matched by lexical BM25 keyword search."
    return "Included in the indexed file set."


def _print_search_response(response: Dict[str, Any], query: str) -> None:
    latency = response.get("latency", {})
    telemetry = latency.get("embedding_profiler") or {}
    print(f"\n--- Hybrid Search: {query} ---")
    print(f"Search latency: {latency.get('search_latency_ms', 0.0):.2f} ms")
    if telemetry:
        print(f"Hardware: {_hardware_badge(telemetry)}")
    results = response.get("results", [])
    if not results:
        print("No matching files found.")
        return
    for index, result in enumerate(results, start=1):
        print(f"\n{index}. {result.get('file_path')}")
        print(f"   Match Score (RRF): {result.get('score', 0.0):.6f}")
        print(f"   Match Source: {result.get('match_source', 'unknown')}")
        print(f"   Episodic Context: {_relationship_reason(result)}")
        print(f"   {_hardware_badge(telemetry)}")


def _run_search(query: str, base_dir: str, top_k: int, include_context: bool = True) -> Dict[str, Any]:
    from src.search import hybrid_search

    database = _connect(base_dir)
    graph = _load_graph(base_dir) if include_context else None
    response = hybrid_search(
        query,
        database,
        top_k=top_k,
        episodic_graph=graph,
        include_episodic_context=False,
    )
    if include_context and graph is not None and response.get("results"):
        seed_file = response["results"][0].get("file_path")
        response = hybrid_search(
            query,
            database,
            top_k=top_k,
            episodic_graph=graph,
            seed_file=seed_file,
            time_window_seconds=DEFAULT_TIME_WINDOW_SECONDS,
        )
    return response


def run_demo() -> None:
    from generate_fixtures import run_seed

    seeded = run_seed(time_window_seconds=DEFAULT_TIME_WINDOW_SECONDS)
    base_dir = seeded["base_dir"]
    print(f"\nDemo vault indexed at: {base_dir}")
    for query in (
        "Q3 planning pipeline dashboard",
        "event pipeline retry queue latency",
        "invoice planning workshop",
    ):
        _print_search_response(_run_search(query, base_dir, top_k=5), query)


def run_profile(model_path: str, tokenizer_path: Optional[str], iterations: int) -> None:
    try:
        from src.features import SnapdragonEmbedder

        embedder = SnapdragonEmbedder(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )
        embedder.profiler.iterations = iterations
        tokenized = embedder._tokenize(["profile the episodic embedding pipeline"])
        sample_input = embedder._session_inputs(tokenized)
        comparison = embedder.profiler.benchmark_provider_comparison(model_path, sample_input)
    except Exception as exc:
        print(f"Unable to run ONNX A/B benchmark: {exc}")
        print("Provide a local ONNX model and tokenizer, for example --model-path models/embedding_model.onnx.")
        return

    print("\n--- Qualcomm QNN vs CPU Inference Profile ---")
    for label in ("qnn", "cpu"):
        result = comparison.get(label)
        if result:
            print(f"{label.upper():4} | {result['latency_ms']:.2f} ms | {result['embeddings_per_second']:.2f} embeddings/sec")
        else:
            print(f"{label.upper():4} | unavailable: {comparison.get(label + '_error', 'provider unavailable')}")
    print(f"Speedup factor: {comparison.get('speedup_factor') or 'N/A'}")
    print(f"Latency delta: {comparison.get('latency_delta_ms') or 'N/A'} ms")
    print(embedder.profiler.format_telemetry_report())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EpisodicFS: hybrid on-device episodic retrieval")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Generate, index, and query a synthetic episodic vault")
    demo_parser.set_defaults(handler=lambda args: run_demo())

    search_parser = subparsers.add_parser("search", help="Run hybrid BM25 and Qualcomm vector search")
    search_parser.add_argument("query", help="Natural-language search query")
    search_parser.add_argument("--base-dir", default="episodic_vault")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--no-episodic-context", action="store_true")
    search_parser.set_defaults(
        handler=lambda args: _print_search_response(
            _run_search(args.query, args.base_dir, args.top_k, not args.no_episodic_context),
            args.query,
        )
    )

    profile_parser = subparsers.add_parser("profile", help="Compare QNN HTP and CPU inference")
    profile_parser.add_argument("--model-path", default="models/embedding_model.onnx")
    profile_parser.add_argument("--tokenizer-path", default="models/embedding_model")
    profile_parser.add_argument("--iterations", type=int, default=10)
    profile_parser.set_defaults(handler=lambda args: run_profile(args.model_path, args.tokenizer_path, args.iterations))

    ingest_parser = subparsers.add_parser("ingest", help="Index an existing vault")
    ingest_parser.add_argument("--base-dir", default="episodic_vault")
    ingest_parser.add_argument("--time-window-hours", type=float, default=1.0)
    ingest_parser.set_defaults(handler=lambda args: _ingest(args))

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        _print_command_guide()
        return 0
    try:
        args.handler(args)
    except (RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


def _ingest(args) -> None:
    from src.db import setup_and_ingest_lancedb

    setup_and_ingest_lancedb(
        base_dir=args.base_dir,
        time_window_seconds=int(args.time_window_hours * 3600),
    )
    print("Ingestion complete.")


def _print_command_guide() -> None:
    print(
        """
EpisodicFS command guide
========================

Demo:       python main.py demo
Search:     python main.py search "event pipeline retry queue latency"
Profile:    python main.py profile --model-path models/embedding_model.onnx
Ingest:     python main.py ingest --base-dir episodic_vault

The demo creates a temporary fixture vault, indexes it, and runs three
episodic hybrid-search examples. Search combines BM25, embeddings, and
temporal-semantic graph context. Profile compares QNN Hexagon HTP with CPU.
Use `python main.py --help` for all options.
""".strip()
    )


if __name__ == "__main__":
    raise SystemExit(main())
