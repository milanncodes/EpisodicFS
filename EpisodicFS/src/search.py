"""Hybrid lexical, dense, and episodic search."""

import math
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .db import open_vault_table
from .features import model
from .graph import EpisodicKnowledgeGraph


_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def _tokens(value: Any) -> List[str]:
    return _TOKEN_PATTERN.findall(str(value or "").lower())


def _record_text(record: Dict[str, Any]) -> str:
    return " ".join(
        str(record.get(field) or "")
        for field in ("file_path", "filename", "file_name", "text_content")
    )


def lexical_search(records: Iterable[Dict[str, Any]], query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Rank indexed records with a small BM25 implementation."""
    records = list(records)
    query_terms = _tokens(query_text)
    if not records or not query_terms:
        return []

    documents = [_tokens(_record_text(record)) for record in records]
    document_frequency = Counter(
        term for document in documents for term in set(document)
    )
    average_length = sum(len(document) for document in documents) / len(documents) or 1.0
    result = []
    for record, document in zip(records, documents):
        term_counts = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = term_counts[term]
            if not frequency:
                continue
            document_count = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (len(records) - document_count + 0.5) / (document_count + 0.5)
            )
            length_factor = 1.2 * (1.0 - 0.75 + 0.75 * len(document) / average_length)
            score += inverse_frequency * frequency * 2.2 / (frequency + length_factor)
        if score > 0:
            result.append(
                {
                    **record,
                    "file_path": record.get("file_path"),
                    "score": score,
                    "match_source": "keyword",
                }
            )
    return sorted(result, key=lambda item: item["score"], reverse=True)[:limit]


def _dense_search(table: Any, records: List[Dict[str, Any]], query_text: str, limit: int) -> List[Dict[str, Any]]:
    query_embedding = model.encode(query_text, convert_to_numpy=True).tolist()
    dense_frame = table.search(query_embedding).metric("cosine").limit(limit).to_pandas()
    dense_results = dense_frame.to_dict(orient="records")
    record_by_path = {record.get("file_path"): record for record in records}
    results = []
    for result in dense_results:
        record = {**record_by_path.get(result.get("file_path"), {}), **result}
        distance = result.get("_distance")
        record["score"] = 1.0 - float(distance) if distance is not None else 0.0
        record["match_source"] = "vector"
        results.append(record)
    return results


def rrf_combine(
    dense_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Merge ranked result lists using Reciprocal Rank Fusion."""
    if k < 1:
        raise ValueError("k must be at least 1")
    combined: Dict[str, Dict[str, Any]] = {}
    for results, source in ((dense_results, "vector"), (lexical_results, "keyword")):
        for rank, result in enumerate(results, start=1):
            file_path = result.get("file_path")
            if not file_path:
                continue
            entry = combined.setdefault(file_path, {**result, "score": 0.0, "_sources": set()})
            entry["score"] += 1.0 / (k + rank)
            entry["_sources"].add(source)
            for key, value in result.items():
                if key not in {"score", "match_source"}:
                    entry.setdefault(key, value)
    output = []
    for entry in sorted(combined.values(), key=lambda item: item["score"], reverse=True):
        sources = entry.pop("_sources")
        entry["match_source"] = "+".join(source for source in ("vector", "keyword") if source in sources)
        output.append(entry)
    return output


def _profiler_metrics() -> Dict[str, Any]:
    embedder = getattr(model, "_instance", None)
    if embedder is None:
        return {}
    return embedder.profiler.format_telemetry_report(as_dict=True)


def _apply_episodic_context(
    results: List[Dict[str, Any]],
    graph: Optional[EpisodicKnowledgeGraph],
    seed_file: Optional[str],
    max_hops: int = 2,
    time_window_seconds: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if graph is None or not seed_file:
        return results
    context = graph.find_episodic_context(
        seed_file,
        max_hops=max_hops,
        time_window_seconds=time_window_seconds,
    )
    result_by_path = {result.get("file_path"): result for result in results}
    maximum_score = results[0]["score"] if results else 0.0
    for item in context:
        file_path = item.get("file_path")
        if not file_path:
            continue
        boost = maximum_score * 0.15 / max(1, item["hops"])
        if file_path in result_by_path:
            result = result_by_path[file_path]
            result["score"] += boost
            result["match_source"] = "+".join(
                dict.fromkeys([result.get("match_source", ""), "episodic"])
            ).strip("+")
        else:
            result = {
                **item,
                "score": boost,
                "match_source": "episodic",
                "episodic_hops": item["hops"],
            }
            results.append(result)
    return sorted(results, key=lambda item: item["score"], reverse=True)


def hybrid_search(
    query_text: str,
    db_connection: Any,
    top_k: int = 5,
    rrf_k: int = 60,
    seed_file: Optional[str] = None,
    time_window_seconds: Optional[int] = None,
    episodic_graph: Optional[EpisodicKnowledgeGraph] = None,
    include_episodic_context: bool = True,
) -> Dict[str, Any]:
    """Return fused vector/keyword results with optional episodic context."""
    started = time.perf_counter()
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    table = open_vault_table(db_connection)
    records = table.to_pandas().to_dict(orient="records")
    dense_started = time.perf_counter()
    dense_results = _dense_search(table, records, query_text, top_k * 2)
    dense_latency_ms = (time.perf_counter() - dense_started) * 1000.0
    lexical_started = time.perf_counter()
    lexical_results = lexical_search(records, query_text, limit=top_k * 2)
    lexical_latency_ms = (time.perf_counter() - lexical_started) * 1000.0
    results = rrf_combine(dense_results, lexical_results, k=rrf_k)[:top_k]

    if include_episodic_context:
        results = _apply_episodic_context(
            results,
            episodic_graph,
            seed_file,
            time_window_seconds=time_window_seconds,
        )[:top_k]
    search_latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "results": [
            {
                "file_path": result.get("file_path"),
                "score": result.get("score", 0.0),
                "match_source": result.get("match_source"),
                **({"record": result} if "record" not in result else {}),
            }
            for result in results
        ],
        "latency": {
            "search_latency_ms": search_latency_ms,
            "dense_latency_ms": dense_latency_ms,
            "lexical_latency_ms": lexical_latency_ms,
            "embedding_profiler": _profiler_metrics(),
        },
    }


def episodic_search(query_text: str, db_connection: Any, top_k: int = 3, include_episodic_context: bool = True):
    """Compatibility wrapper returning the original three-value search tuple."""
    response = hybrid_search(
        query_text,
        db_connection,
        top_k=top_k,
        include_episodic_context=include_episodic_context,
    )
    direct_hits = [item for item in response["results"] if "vector" in item["match_source"]]
    context_hits = [item for item in response["results"] if item not in direct_hits]
    return direct_hits, context_hits, response["latency"]["search_latency_ms"]


def display_results(query_text: str, direct_hits: List[Dict[str, Any]], episodic_context_hits: List[Dict[str, Any]], total_time_ms: float):
    """Print structured hybrid results in the existing CLI format."""
    print(f"\n--- Search Results for: '{query_text}' (Query Time: {total_time_ms:.2f} ms) ---")
    all_results = direct_hits + episodic_context_hits
    if not all_results:
        print("No matching files found.")
        return
    for index, result in enumerate(all_results, start=1):
        print(f"\nRank {index} ({result.get('match_source', 'unknown')}):")
        print(f"  File Path: {result.get('file_path')}")
        print(f"  Score: {result.get('score', 0.0):.6f}")
